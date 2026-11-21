// Inkling 2 Mini の GPU カーネル。1トークンずつ処理する前提で、行列は全部 行列×ベクトル。
// 計算の順序と式は src/model.rs の CPU 版と同じ。

#define D 2304
#define HD 128
#define NREL 16
#define EXTENT 1024
#define FF 1152
#define TOPK 8
#define INFINITY __int_as_float(0x7f800000)

typedef unsigned short bf16;
typedef unsigned char u8;

__device__ __forceinline__ float bf(unsigned x) { return __uint_as_float(x << 16); }

// E2M1 と E4M3 は、ビットを f32 の指数部と仮数部の位置へずらすと 2^-126 倍、2^-120 倍の値になる (非正規化数も同じ式で合う)
__device__ __forceinline__ float e2m1(unsigned n) { return __uint_as_float(((n & 7u) << 22) | ((n & 8u) << 28)) * 0x1p126f; }
__device__ __forceinline__ float e4m3(unsigned b) { return __uint_as_float(((b & 0x7fu) << 20) | ((b & 0x80u) << 24)) * 0x1p120f; }
__device__ __forceinline__ float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }

__device__ __forceinline__ float warp_sum(float s) {
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    return s;
}

__device__ __forceinline__ float warp_max(float s) {
    for (int o = 16; o > 0; o >>= 1) s = fmaxf(s, __shfl_xor_sync(0xffffffffu, s, o));
    return s;
}

// blockDim.x は 32 の倍数
__device__ float block_sum(float s) {
    __shared__ float sh[32];
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    s = warp_sum(s);
    __syncthreads();
    if (lane == 0) sh[w] = s;
    __syncthreads();
    s = threadIdx.x < (blockDim.x >> 5) ? sh[threadIdx.x] : 0.f;
    if (w == 0) s = warp_sum(s);
    if (threadIdx.x == 0) sh[0] = s;
    __syncthreads();
    return sh[0];
}

__device__ float block_max(float s) {
    __shared__ float sh[32];
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    s = warp_max(s);
    __syncthreads();
    if (lane == 0) sh[w] = s;
    __syncthreads();
    s = threadIdx.x < (blockDim.x >> 5) ? sh[threadIdx.x] : -INFINITY;
    if (w == 0) s = warp_max(s);
    if (threadIdx.x == 0) sh[0] = s;
    __syncthreads();
    return sh[0];
}

// y = W x。W は BF16 の [rows, cols]、cols は 8 の倍数。1行を1ワープで計算する
extern "C" __global__ void mv_bf16(const bf16* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, int rows, int cols) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= rows) return;
    const uint4* wr = (const uint4*)(w + (size_t)row * cols);
    float s = 0.f;
    for (int i = lane; i < cols / 8; i += 32) {
        uint4 v = __ldg(wr + i);
        const float4* xx = (const float4*)(x + i * 8);
        float4 a = xx[0], b = xx[1];
        s += bf(v.x & 0xffffu) * a.x + bf(v.x >> 16) * a.y + bf(v.y & 0xffffu) * a.z + bf(v.y >> 16) * a.w;
        s += bf(v.z & 0xffffu) * b.x + bf(v.z >> 16) * b.y + bf(v.w & 0xffffu) * b.z + bf(v.w >> 16) * b.w;
    }
    s = warp_sum(s);
    if (lane == 0) y[row] = s;
}

// count 本の長さ n のベクトルをそれぞれ RMS 正規化する。w が 0 なら重みなし、wstride が 0 なら全ベクトルで同じ重み
extern "C" __global__ void rmsnorm(const float* x, const float* w, float* y, int n, int wstride) {
    x += blockIdx.x * n;
    y += blockIdx.x * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += x[i] * x[i];
    float k = rsqrtf(block_sum(ss) / n + 1e-6f);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * k * (w ? w[blockIdx.x * wstride + i] : 1.f);
}

// 深さ方向の注意。状態は blocks[0..nb] と cur。query 1本につき1ブロック
extern "C" __global__ void mix(const float* blocks, int nb, const float* cur, const float* queries, float* out) {
    const float* q = queries + blockIdx.x * D;
    float sc[8], inv[8];
    int ns = nb + 1;
    for (int s = 0; s < ns; s++) {
        const float* st = s < nb ? blocks + s * D : cur;
        float ss = 0.f, dq = 0.f;
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            ss += st[i] * st[i];
            dq += st[i] * q[i];
        }
        ss = block_sum(ss);
        dq = block_sum(dq);
        inv[s] = rsqrtf(ss / D + 1e-6f);
        sc[s] = dq * inv[s] * rsqrtf((float)D);
    }
    float m = -INFINITY, z = 0.f;
    for (int s = 0; s < ns; s++) m = fmaxf(m, sc[s]);
    for (int s = 0; s < ns; s++) z += sc[s] = expf(sc[s] - m);
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float o = 0.f;
        for (int s = 0; s < ns; s++) o += sc[s] / z * inv[s] * (s < nb ? blocks + s * D : cur)[i];
        out[blockIdx.x * D + i] = o;
    }
}

// control の前半: [rmsnorm(cur), rmsnorm(msg)] を out に並べる。2ブロック
extern "C" __global__ void norm2(const float* cur, const float* msg, float* out) {
    const float* x = blockIdx.x ? msg : cur;
    float ss = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) ss += x[i] * x[i];
    float k = rsqrtf(block_sum(ss) / D + 1e-6f);
    for (int i = threadIdx.x; i < D; i += blockDim.x) out[blockIdx.x * D + i] = x[i] * k;
}

// control の後半: gf は [gain, forget]。cur を上書きしてよい
extern "C" __global__ void control(float* cur, const float* msg, const float* gf) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= D) return;
    float c = cur[i];
    cur[i] = c - tanhf(gf[D + i]) * c + tanhf(gf[i]) * msg[i];
}

// 因果的な短い畳み込みと残差。w は [ch][4] で k=3 が現在、dyn はヘッドごとの [lag] (0 なら使わない)、hist は [ch][3]
extern "C" __global__ void conv(const float* cur, const float* w, const float* dyn, float* hist, float* out, int ch, int cph) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    float h0 = hist[c * 3], h1 = hist[c * 3 + 1], h2 = hist[c * 3 + 2], x = cur[c];
    float o = w[c * 4] * h0 + w[c * 4 + 1] * h1 + w[c * 4 + 2] * h2 + w[c * 4 + 3] * x;
    if (dyn) {
        const float* d = dyn + (c / cph) * 4;
        o += d[0] * x + d[1] * h2 + d[2] * h1 + d[3] * h0;
    }
    hist[c * 3] = h1;
    hist[c * 3 + 1] = h2;
    hist[c * 3 + 2] = x;
    out[c] = (x + o) * (dyn ? 1.f / 3.f : rsqrtf(5.f));
}

__device__ float clip_scale(float v, float threshold) {
    float norm = sqrtf(block_sum(v * v)) + 1e-6f;
    return fminf(threshold / norm, 1.f);
}

// Q, R, K の L2 クリップ、V の正規化、global 層の tau。K と V はキャッシュの行に書く。64ブロック x 128スレッド
extern "C" __global__ void qkv_post(float* q, float* r, const float* k, const float* v, const float* qclip, const float* rclip,
                                    float tau, float* kc, float* vc) {
    int b = blockIdx.x, t = threadIdx.x;
    if (b < 24) {
        float x = q[b * HD + t] * qclip[b * HD + t];
        x *= clip_scale(x, 22.627416997969522f);
        q[b * HD + t] = x * tau;
    } else if (b < 48) {
        int h = b - 24;
        float x = t < NREL ? r[h * NREL + t] * rclip[h * NREL + t] : 0.f;
        x *= clip_scale(x, 8.f);
        if (t < NREL) r[h * NREL + t] = x * tau;
    } else if (b < 56) {
        int h = b - 48;
        float x = k[h * HD + t];
        kc[h * HD + t] = x * clip_scale(x, 22.627416997969522f);
    } else {
        int h = b - 56;
        float x = v[h * HD + t];
        vc[h * HD + t] = x * rsqrtf(block_sum(x * x) / HD + 1e-6f);
    }
}

// 1ヘッド1ブロック、128スレッド。位置 first..=pos を見る。ring なら SWA のリングバッファ (位置 % EXTENT)
// 共有メモリ: 相対位置バイアス EXTENT 個 + スコア (pos + 1 - first) 個
extern "C" __global__ void attn(const float* q, const float* r, const float* sinks, const float* kc, const float* vc, const float* rel_proj,
                                const float* gate, float* out, int pos, int first, int ring) {
    extern __shared__ float sm[];
    float* rel = sm;
    float* sc = sm + EXTENT;
    int h = blockIdx.x, kvh = h / 3, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    float rh[NREL];
    for (int i = 0; i < NREL; i++) rh[i] = r[h * NREL + i];
    for (int d = t; d < EXTENT; d += blockDim.x) {
        float s = 0.f;
        for (int i = 0; i < NREL; i++) s += rh[i] * rel_proj[i * EXTENT + d];
        rel[d] = s;
    }
    __syncthreads();
    float4 qv = ((const float4*)(q + h * HD))[lane];
    int n = pos + 1;
    for (int j = first + warp; j < n; j += blockDim.x >> 5) {
        int idx = ring ? j % EXTENT : j;
        float4 kv = ((const float4*)(kc + (size_t)idx * 1024 + kvh * HD))[lane];
        float s = warp_sum(qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w);
        int d = pos - j;
        if (lane == 0) sc[j - first] = s * rsqrtf((float)HD) + (d < EXTENT ? rel[d] : 0.f);
    }
    __syncthreads();
    // sink のロジットは sinks[h]、値はゼロベクトル
    float m = sinks[h];
    for (int i = t; i < n - first; i += blockDim.x) m = fmaxf(m, sc[i]);
    m = block_max(m);
    float z = t == 0 ? expf(sinks[h] - m) : 0.f;
    for (int i = t; i < n - first; i += blockDim.x) z += sc[i] = expf(sc[i] - m);
    z = block_sum(z);
    float o = 0.f;
    for (int j = first; j < n; j++) o += sc[j - first] * vc[(size_t)(ring ? j % EXTENT : j) * 1024 + kvh * HD + t];
    out[h * HD + t] = o / z * sigmoidf(gate[h * HD + t]);
}

extern "C" __global__ void add(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

extern "C" __global__ void softcap(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 30.f * tanhf(x[i] / 30.f);
}

// ルーター: sigmoid、バイアス込みで上位8個を選び、重みを正規化して route_scale (8) を掛ける。1ブロック x 128スレッド
extern "C" __global__ void topk(const float* logits, const float* bias, int* ids, float* wts) {
    __shared__ float p[128], s[128];
    int t = threadIdx.x;
    p[t] = sigmoidf(logits[t]);
    s[t] = p[t] + bias[t];
    __syncthreads();
    if (t == 0) {
        float sum = 0.f;
        for (int k = 0; k < TOPK; k++) {
            int bi = 0;
            for (int e = 1; e < 128; e++)
                if (s[e] > s[bi]) bi = e;
            ids[k] = bi;
            wts[k] = p[bi];
            sum += p[bi];
            s[bi] = -INFINITY;
        }
        for (int k = 0; k < TOPK; k++) wts[k] = wts[k] / (sum + 1e-20f) * 8.f;
    }
}

// NVFP4 の1行 (n 要素) と x の内積の部分和。ワープの各レーンが 16 要素ブロックを分担する
__device__ float nvfp4_row(const u8* w, const u8* s, const float* x, int n, int lane) {
    float acc = 0.f;
    for (int blk = lane; blk < n / 16; blk += 32) {
        uint2 pk = *(const uint2*)(w + blk * 8);
        const float4* xx = (const float4*)(x + blk * 16);
        float b = 0.f;
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            unsigned u = half ? pk.y : pk.x;
            float4 x0 = xx[half * 2], x1 = xx[half * 2 + 1];
            b += e2m1(u) * x0.x + e2m1(u >> 4) * x0.y + e2m1(u >> 8) * x0.z + e2m1(u >> 12) * x0.w;
            b += e2m1(u >> 16) * x1.x + e2m1(u >> 20) * x1.y + e2m1(u >> 24) * x1.z + e2m1(u >> 28) * x1.w;
        }
        acc += b * e4m3(s[blk]);
    }
    return acc;
}

// 選ばれた8 expert の gate と up を計算して SITU をかける。1ワープが gate の行 2i と up の行 2i+1 を受け持つ
extern "C" __global__ void moe_up_nvfp4(const u8* w13, const u8* s13, const float* a13, const int* ids, const float* x, float* act) {
    int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (gw >= TOPK * FF) return;
    int k = gw / FF, i = gw % FF, ex = ids[k];
    size_t row = (size_t)ex * 2 * FF + 2 * i;
    float g = warp_sum(nvfp4_row(w13 + row * (D / 2), s13 + row * (D / 16), x, D, lane));
    float u = warp_sum(nvfp4_row(w13 + (row + 1) * (D / 2), s13 + (row + 1) * (D / 16), x, D, lane));
    float a = a13[ex] / 2688.f;
    g *= a;
    u *= a;
    if (lane == 0) act[k * FF + i] = 4.f * tanhf(g / 4.f) * sigmoidf(g) * 25.f * tanhf(u / 25.f);
}

// down 射影と8 expert の重み付き和。1ワープが出力の1行を受け持つ
extern "C" __global__ void moe_down_nvfp4(const u8* w2, const u8* s2, const float* a2, const int* ids, const float* wts, const float* act, float* y) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= D) return;
    float acc = 0.f;
    for (int k = 0; k < TOPK; k++) {
        int ex = ids[k];
        size_t r = (size_t)ex * D + row;
        acc += nvfp4_row(w2 + r * (FF / 2), s2 + r * (FF / 16), act + k * FF, FF, lane) * wts[k] * a2[ex] / 2688.f;
    }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}
