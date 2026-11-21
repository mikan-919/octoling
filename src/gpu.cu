// Inkling 2 Mini の GPU カーネル。T 個のトークンをまとめて処理する (生成中は T = 1、prefill では T <= TMAX)。
// 計算の順序と式は src/model.rs の CPU 版と同じ。
// バッファの並びは [種類][T][要素]。たとえば blocks は [ブロック番号][T][D]、inp は [Q/K/V][T][D]。

#define D 2304
#define HD 128
#define NREL 16
#define EXTENT 1024
#define FF 1152
#define TOPK 8
#define NEXP 128
#define QX 6552 // [q 3072, r 384, gate 3072, sink 24]
#define R_OFF 3072
#define GATE_OFF 3456
#define SINK_OFF 6528
#define INFINITY __int_as_float(0x7f800000)

typedef unsigned short bf16;
typedef unsigned char u8;

__device__ __forceinline__ float bf(unsigned x) { return __uint_as_float(x << 16); }

// E2M1 と E4M3 は、ビットを f32 の指数部と仮数部の位置へずらすと 2^-126 倍、2^-120 倍の値になる (非正規化数も同じ式で合う)
__device__ __forceinline__ float e2m1(unsigned n) { return __uint_as_float(((n & 7u) << 22) | ((n & 8u) << 28)) * 0x1p126f; }
__device__ __forceinline__ float e4m3(unsigned b) { return __uint_as_float(((b & 0x7fu) << 20) | ((b & 0x80u) << 24)) * 0x1p120f; }
__device__ __forceinline__ float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ float situ(float g, float u) { return 4.f * tanhf(g / 4.f) * sigmoidf(g) * 25.f * tanhf(u / 25.f); }

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

// y = W x (T = 1 用)。W は BF16 の [rows, cols]、cols は 8 の倍数。1行を1ワープで計算する
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

// 64行 x 32列のタイルの積和。Ws[k][行]、Xs[k][列] を共有メモリに置き、各スレッドが 4行 x 2列を受け持つ
#define TILE_MAC(acc)                                                         \
    for (int kk = 0; kk < 32; kk++) {                                         \
        float4 w4 = *(const float4*)&Ws[kk][ty * 4];                          \
        float2 x2 = *(const float2*)&Xs[kk][tx * 2];                          \
        acc[0][0] += w4.x * x2.x; acc[0][1] += w4.x * x2.y;                   \
        acc[1][0] += w4.y * x2.x; acc[1][1] += w4.y * x2.y;                   \
        acc[2][0] += w4.z * x2.x; acc[2][1] += w4.z * x2.y;                   \
        acc[3][0] += w4.w * x2.x; acc[3][1] += w4.w * x2.y;                   \
    }

// Y[t][r] = sum_c W[r][c] X[t][c]。W は BF16 の [rows, cols]、cols は 32 の倍数。ブロックは 64行 x 32トークン、256スレッド
extern "C" __global__ void gemm_bf16(const bf16* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, int rows, int cols, int T) {
    __shared__ __align__(16) float Ws[32][64];
    __shared__ __align__(16) float Xs[32][32];
    int tid = threadIdx.x, ty = tid / 16, tx = tid % 16;
    int r0 = blockIdx.x * 64, t0 = blockIdx.y * 32;
    float acc[4][2] = {};
    for (int k0 = 0; k0 < cols; k0 += 32) {
        // W のタイル: 64行 x 32列 = 2048 要素、1スレッド 8 要素 (uint4)
        {
            int r = tid / 4, c = (tid % 4) * 8;
            uint4 v = r0 + r < rows ? __ldg((const uint4*)(w + (size_t)(r0 + r) * cols + k0 + c)) : make_uint4(0, 0, 0, 0);
            unsigned u[4] = {v.x, v.y, v.z, v.w};
            for (int i = 0; i < 4; i++) {
                Ws[c + 2 * i][r] = bf(u[i] & 0xffffu);
                Ws[c + 2 * i + 1][r] = bf(u[i] >> 16);
            }
        }
        // X のタイル: 32トークン x 32列、1スレッド 4 要素
        {
            int j = tid / 8, c = (tid % 8) * 4;
            float4 v = t0 + j < T ? *(const float4*)(x + (size_t)(t0 + j) * cols + k0 + c) : make_float4(0, 0, 0, 0);
            Xs[c][j] = v.x; Xs[c + 1][j] = v.y; Xs[c + 2][j] = v.z; Xs[c + 3][j] = v.w;
        }
        __syncthreads();
        TILE_MAC(acc)
        __syncthreads();
    }
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 2; j++) {
            int r = r0 + ty * 4 + i, t = t0 + tx * 2 + j;
            if (r < rows && t < T) y[(size_t)t * rows + r] = acc[i][j];
        }
}

// 長さ n のベクトル gridDim.x 本をそれぞれ RMS 正規化する。並びは [種類][T][n] で、重みは種類ごとに wstride ずつずれる (0 なら共通)
extern "C" __global__ void rmsnorm(const float* x, const float* w, float* y, int n, int wstride, int T) {
    x += (size_t)blockIdx.x * n;
    y += (size_t)blockIdx.x * n;
    if (w) w += (blockIdx.x / T) * wstride;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += x[i] * x[i];
    float k = rsqrtf(block_sum(ss) / n + 1e-6f);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * k * (w ? w[i] : 1.f);
}

// 深さ方向の注意。状態は blocks[0..nb] と cur。ブロックは (query, トークン)。out は [query][T][D]
extern "C" __global__ void mix(const float* blocks, int nb, const float* cur, const float* queries, float* out, int T) {
    int t = blockIdx.y;
    const float* q = queries + blockIdx.x * D;
    float sc[8], inv[8];
    int ns = nb + 1;
    for (int s = 0; s < ns; s++) {
        const float* st = s < nb ? blocks + ((size_t)s * T + t) * D : cur + (size_t)t * D;
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
        for (int s = 0; s < ns; s++) o += sc[s] / z * inv[s] * (s < nb ? blocks + ((size_t)s * T + t) * D : cur + (size_t)t * D)[i];
        out[((size_t)blockIdx.x * T + t) * D + i] = o;
    }
}

// control の前半: トークンごとに [rmsnorm(cur), rmsnorm(msg)] を out [T][2D] に並べる。ブロックは (2, T)
extern "C" __global__ void norm2(const float* cur, const float* msg, float* out) {
    int t = blockIdx.y;
    const float* x = (blockIdx.x ? msg : cur) + (size_t)t * D;
    float ss = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) ss += x[i] * x[i];
    float k = rsqrtf(block_sum(ss) / D + 1e-6f);
    for (int i = threadIdx.x; i < D; i += blockDim.x) out[(size_t)t * 2 * D + blockIdx.x * D + i] = x[i] * k;
}

// control の後半: gf は [T][gain D, forget D]。cur を上書きしてよい。ブロックは (D/256, T)
extern "C" __global__ void control(float* cur, const float* msg, const float* gf) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= D) return;
    size_t j = (size_t)t * D + i;
    const float* g = gf + (size_t)t * 2 * D;
    float c = cur[j];
    cur[j] = c - tanhf(g[D + i]) * c + tanhf(g[i]) * msg[j];
}

// 因果的な短い畳み込みと残差。トークンを順に流す。w は [ch][4] で k=3 が現在、hist は [ch][3]
// 入力の行は cs 要素おき、出力の行は os 要素おき。dyn >= 0 なら入力の行の dyn 番目からがヘッドごとの [lag] のタップ
extern "C" __global__ void conv(const float* cur, int cs, const float* w, int dyn, float* hist, float* out, int os, int ch, int cph, int T) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    float h0 = hist[c * 3], h1 = hist[c * 3 + 1], h2 = hist[c * 3 + 2];
    float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int t = 0; t < T; t++) {
        const float* row = cur + (size_t)t * cs;
        float x = row[c];
        float o = w0 * h0 + w1 * h1 + w2 * h2 + w3 * x;
        if (dyn >= 0) {
            const float* d = row + dyn + (c / cph) * 4;
            o += d[0] * x + d[1] * h2 + d[2] * h1 + d[3] * h0;
        }
        out[(size_t)t * os + c] = (x + o) * (dyn >= 0 ? 1.f / 3.f : rsqrtf(5.f));
        h0 = h1;
        h1 = h2;
        h2 = x;
    }
    hist[c * 3] = h0;
    hist[c * 3 + 1] = h1;
    hist[c * 3 + 2] = h2;
}

__device__ float clip_scale(float v, float threshold) {
    float norm = sqrtf(block_sum(v * v)) + 1e-6f;
    return fminf(threshold / norm, 1.f);
}

// Q, R, K の L2 クリップ、V の正規化、global 層の tau。K と V はキャッシュの行に書く。ブロックは (64, T) x 128スレッド
// 位置はデバイスのメモリから読む (先頭トークンの位置)。CUDA Graph で同じカーネル列を毎トークン再生するため
// ring > 0 なら SWA で、キャッシュは ring 行のリングバッファ
extern "C" __global__ void qkv_post(float* qx, const float* kp, const float* vp, const float* qclip, const float* rclip, const int* posp,
                                    int ring, float* kc, float* vc) {
    int b = blockIdx.x, t = threadIdx.x, pos = *posp + blockIdx.y;
    float* q = qx + (size_t)blockIdx.y * QX;
    float* r = q + R_OFF;
    const float* k = kp + (size_t)blockIdx.y * 1024;
    const float* v = vp + (size_t)blockIdx.y * 1024;
    float tau = ring ? 1.f : 1.f + 0.1f * logf(fmaxf(1.f, (pos + 1.f) / 8192.f));
    size_t row = (size_t)(ring ? pos % ring : pos) * 1024;
    kc += row;
    vc += row;
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

// ブロックは (ヘッド, トークン)、128スレッド。ring > 0 なら SWA で、直近 EXTENT 個の位置をリングバッファ (位置 % ring) から読む
// 共有メモリ: 相対位置バイアス EXTENT 個 + スコア (pos + 1 - first) 個
extern "C" __global__ void attn(const float* qx, const float* kc, const float* vc, const float* rel_proj, float* out, const int* posp, int ring) {
    extern __shared__ float sm[];
    float* rel = sm;
    float* sc = sm + EXTENT;
    int pos = *posp + blockIdx.y, first = ring ? max(0, pos + 1 - EXTENT) : 0;
    const float* q = qx + (size_t)blockIdx.y * QX;
    int h = blockIdx.x, kvh = h / 3, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    float rh[NREL];
    for (int i = 0; i < NREL; i++) rh[i] = q[R_OFF + h * NREL + i];
    for (int d = t; d < EXTENT; d += blockDim.x) {
        float s = 0.f;
        for (int i = 0; i < NREL; i++) s += rh[i] * rel_proj[i * EXTENT + d];
        rel[d] = s;
    }
    __syncthreads();
    float4 qv = ((const float4*)(q + h * HD))[lane];
    int n = pos + 1;
    for (int j = first + warp; j < n; j += blockDim.x >> 5) {
        int idx = ring ? j % ring : j;
        float4 kv = ((const float4*)(kc + (size_t)idx * 1024 + kvh * HD))[lane];
        float s = warp_sum(qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w);
        int d = pos - j;
        if (lane == 0) sc[j - first] = s * rsqrtf((float)HD) + (d < EXTENT ? rel[d] : 0.f);
    }
    __syncthreads();
    // sink のロジットは sinks[h]、値はゼロベクトル
    float sink = q[SINK_OFF + h];
    float m = sink;
    for (int i = t; i < n - first; i += blockDim.x) m = fmaxf(m, sc[i]);
    m = block_max(m);
    float z = t == 0 ? expf(sink - m) : 0.f;
    for (int i = t; i < n - first; i += blockDim.x) z += sc[i] = expf(sc[i] - m);
    z = block_sum(z);
    float o = 0.f;
    for (int j = first; j < n; j++) o += sc[j - first] * vc[(size_t)(ring ? j % ring : j) * 1024 + kvh * HD + t];
    out[(size_t)blockIdx.y * 3072 + h * HD + t] = o / z * sigmoidf(q[GATE_OFF + h * HD + t]);
}

extern "C" __global__ void add(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

extern "C" __global__ void softcap(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 30.f * tanhf(x[i] / 30.f);
}

// ルーター: sigmoid、バイアス込みで上位8個を選び、重みを正規化して route_scale (8) を掛ける。ブロックはトークンごと、128スレッド
extern "C" __global__ void topk(const float* logits, const float* bias, int* ids, float* wts) {
    __shared__ float p[NEXP], s[NEXP];
    int t = threadIdx.x;
    logits += blockIdx.x * NEXP;
    ids += blockIdx.x * TOPK;
    wts += blockIdx.x * TOPK;
    p[t] = sigmoidf(logits[t]);
    s[t] = p[t] + bias[t];
    __syncthreads();
    if (t == 0) {
        float sum = 0.f;
        for (int k = 0; k < TOPK; k++) {
            int bi = 0;
            for (int e = 1; e < NEXP; e++)
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

// (T = 1 用) 選ばれた8 expert の gate と up を計算して SITU をかける。1ワープが gate の行 2i と up の行 2i+1 を受け持つ
extern "C" __global__ void moe_up_nvfp4(const u8* w13, const u8* s13, const float* a13, const int* ids, const float* x, float* act) {
    int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (gw >= TOPK * FF) return;
    int k = gw / FF, i = gw % FF, ex = ids[k];
    size_t row = (size_t)ex * 2 * FF + 2 * i;
    float g = warp_sum(nvfp4_row(w13 + row * (D / 2), s13 + row * (D / 16), x, D, lane));
    float u = warp_sum(nvfp4_row(w13 + (row + 1) * (D / 2), s13 + (row + 1) * (D / 16), x, D, lane));
    float a = a13[ex] / 2688.f;
    if (lane == 0) act[k * FF + i] = situ(g * a, u * a);
}

// (T = 1 用) down 射影と8 expert の重み付き和。1ワープが出力の1行を受け持つ
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

// (prefill 用) T*8 個の割り当て (トークン t の k 番目 = t*8+k) を expert 順に並べる。order[off[e]..off[e+1]] が expert e の割り当て
extern "C" __global__ void moe_sort(const int* ids, int n, int* order, int* off) {
    if (threadIdx.x != 0) return;
    int cnt[NEXP] = {};
    for (int a = 0; a < n; a++) cnt[ids[a]]++;
    off[0] = 0;
    for (int e = 0; e < NEXP; e++) off[e + 1] = off[e] + cnt[e];
    for (int e = 0; e < NEXP; e++) cnt[e] = off[e];
    for (int a = 0; a < n; a++) order[cnt[ids[a]]++] = a;
}

// (prefill 用) expert ごとに、その expert に割り当てられた入力行をまとめて掛ける。重みはバッチにつき1回だけ読む
// Y[a][r] = amax/2688 * sum_c W_e[r][c] X[a / xdiv][c]。ブロックは (rows/64, expert)、256スレッド
extern "C" __global__ void gemm_nvfp4_grouped(const u8* __restrict__ w, const u8* __restrict__ s, const float* amax, int rows, int cols,
                                              const float* __restrict__ x, int xdiv, const int* order, const int* off, float* y) {
    __shared__ __align__(16) float Ws[32][64];
    __shared__ __align__(16) float Xs[32][32];
    __shared__ int as[32];
    int e = blockIdx.y, beg = off[e], end = off[e + 1];
    if (beg == end) return;
    int tid = threadIdx.x, ty = tid / 16, tx = tid % 16, r0 = blockIdx.x * 64;
    const u8* we = w + (size_t)e * rows * (cols / 2);
    const u8* se = s + (size_t)e * rows * (cols / 16);
    float g = amax[e] / 2688.f;
    for (int c0 = beg; c0 < end; c0 += 32) {
        if (tid < 32) as[tid] = c0 + tid < end ? order[c0 + tid] : -1;
        float acc[4][2] = {};
        for (int k0 = 0; k0 < cols; k0 += 32) {
            __syncthreads();
            // W のタイル: 64行 x 32列 = 行あたり 16 バイトとスケール 2 個。1スレッド 8 要素
            {
                int r = tid / 4, c = (tid % 4) * 8;
                const u8* p = we + (size_t)(r0 + r) * (cols / 2) + (k0 + c) / 2;
                unsigned u = *(const unsigned*)p;
                float sc = e4m3(se[(size_t)(r0 + r) * (cols / 16) + (k0 + c) / 16]) * g;
                for (int i = 0; i < 8; i++) Ws[c + i][r] = e2m1(u >> (4 * i)) * sc;
            }
            {
                int j = tid / 8, c = (tid % 8) * 4;
                int a = as[j];
                float4 v = a >= 0 ? *(const float4*)(x + (size_t)(a / xdiv) * cols + k0 + c) : make_float4(0, 0, 0, 0);
                Xs[c][j] = v.x; Xs[c + 1][j] = v.y; Xs[c + 2][j] = v.z; Xs[c + 3][j] = v.w;
            }
            __syncthreads();
            TILE_MAC(acc)
        }
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 2; j++) {
                int a = as[tx * 2 + j];
                if (a >= 0) y[(size_t)a * rows + r0 + ty * 4 + i] = acc[i][j];
            }
        __syncthreads();
    }
}

// (prefill 用) gate/up が交互に並ぶ [n][2FF] に SITU をかけて [n][FF] にする
extern "C" __global__ void situ_rows(const float* gu, float* act, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * FF) return;
    int a = i / FF, j = i % FF;
    act[i] = situ(gu[(size_t)a * 2 * FF + 2 * j], gu[(size_t)a * 2 * FF + 2 * j + 1]);
}

// (prefill 用) y[t] = sum_k wts[t][k] yd[t*8+k]
extern "C" __global__ void moe_sum(const float* yd, const float* wts, float* y, int T) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * D) return;
    int t = i / D, d = i % D;
    float s = 0.f;
    for (int k = 0; k < TOPK; k++) s += wts[t * TOPK + k] * yd[((size_t)t * TOPK + k) * D + d];
    y[i] = s;
}
