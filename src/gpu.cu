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
typedef unsigned short half_t; // KV キャッシュは f16

__device__ __forceinline__ half_t f2h(float f) {
    half_t h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}

__device__ __forceinline__ float h2f(half_t h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

// f16 4つを f32 4つに
__device__ __forceinline__ float4 h4(const half_t* p) {
    uint2 u = *(const uint2*)p;
    return make_float4(h2f(u.x & 0xffff), h2f(u.x >> 16), h2f(u.y & 0xffff), h2f(u.y >> 16));
}

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

// ---- テンソルコアの行列積 (prefill 用) ----
// mma.sync m16n8k16 (BF16 入力、f32 累積) を使う。ブロックは 128スレッド (4ワープ) で 64行 x 32列 (トークン) を受け持ち、
// ワープ w が行 16w..16w+15 と 32列 (8列のタイル4つ) を計算する。K は 32 ずつ共有メモリに載せる。
// 共有メモリの行は 40 個の BF16 (32 + 8 の詰め物) で、フラグメントを読むときにバンクが衝突しない。
#define TS 40

__device__ __forceinline__ unsigned short f2bf(float f) {
    unsigned u = __float_as_uint(f);
    return (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

__device__ __forceinline__ unsigned pack2(float a, float b) { return (unsigned)f2bf(a) | ((unsigned)f2bf(b) << 16); }

// Ws [64][TS]、Xs [2][32][TS] (どちらも K が連続) から、K 32 個分の積和を acc[4][4] に足す。
// Xs は活性を BF16 2つの和 (上位 + 残り) に分けたもので、重みと2回掛けて足す。重みは BF16 で誤差がないので、
// 活性の精度は約16ビットになる (BF16 1つだけだと 8ビットで、48層を通すと logits が 1% ほどずれた)
__device__ __forceinline__ void mma_tile(const unsigned short* Ws, const unsigned short* Xs, float acc[4][4]) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const unsigned* W32 = (const unsigned*)Ws;
    #pragma unroll
    for (int ks = 0; ks < 32; ks += 16) {
        int r = warp * 16 + g, k = ks / 2 + t;
        unsigned a0 = W32[r * (TS / 2) + k], a1 = W32[(r + 8) * (TS / 2) + k];
        unsigned a2 = W32[r * (TS / 2) + k + 4], a3 = W32[(r + 8) * (TS / 2) + k + 4];
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            const unsigned* X32 = (const unsigned*)(Xs + h * 32 * TS);
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                unsigned b0 = X32[(j * 8 + g) * (TS / 2) + k], b1 = X32[(j * 8 + g) * (TS / 2) + k + 4];
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+f"(acc[j][0]), "+f"(acc[j][1]), "+f"(acc[j][2]), "+f"(acc[j][3])
                             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
    }
}

// x を BF16 の上位と残りに分ける
__device__ __forceinline__ void split2(float a, float b, unsigned& hi, unsigned& lo) {
    unsigned short ha = f2bf(a), hb = f2bf(b);
    hi = (unsigned)ha | ((unsigned)hb << 16);
    lo = pack2(a - bf(ha), b - bf(hb));
}

// X の 32 行 x K 32 個を BF16 2つの和にして Xs [2][32][TS] に置く。row(j) が負なら 0 で埋める。1スレッド 8 要素
template <typename F>
__device__ __forceinline__ void load_x(unsigned short* Xs, const float* x, int cols, int k0, F row) {
    int j = threadIdx.x / 4, c = (threadIdx.x % 4) * 8, r = row(j);
    uint4 hi = make_uint4(0, 0, 0, 0), lo = hi;
    if (r >= 0) {
        const float4* p = (const float4*)(x + (size_t)r * cols + k0 + c);
        float4 a = p[0], b = p[1];
        split2(a.x, a.y, hi.x, lo.x);
        split2(a.z, a.w, hi.y, lo.y);
        split2(b.x, b.y, hi.z, lo.z);
        split2(b.z, b.w, hi.w, lo.w);
    }
    *(uint4*)&Xs[j * TS + c] = hi;
    *(uint4*)&Xs[32 * TS + j * TS + c] = lo;
}

// 出力: acc[j] は行 16w+g と 16w+g+8、列 8j+2t と 8j+2t+1
template <typename F>
__device__ __forceinline__ void store_y(float acc[4][4], int r0, F put) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    for (int j = 0; j < 4; j++) {
        int r = r0 + warp * 16 + g, c = j * 8 + 2 * t;
        put(r, c, acc[j][0]);
        put(r, c + 1, acc[j][1]);
        put(r + 8, c, acc[j][2]);
        put(r + 8, c + 1, acc[j][3]);
    }
}

// Y[t][r] = sum_c W[r][c] X[t][c]。W は BF16 の [rows, cols]、cols は 32 の倍数。グリッドは (rows/64, T/32)
extern "C" __global__ void gemm_bf16(const bf16* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, int rows, int cols, int T) {
    __shared__ __align__(16) unsigned short Ws[64 * TS], Xs[2 * 32 * TS];
    int r0 = blockIdx.x * 64, t0 = blockIdx.y * 32, tid = threadIdx.x;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < cols; k0 += 32) {
        // W のタイル: 64行 x 32列。1スレッド 16 要素 (uint4 2つ)
        {
            int r = tid / 2, c = (tid % 2) * 16;
            uint4 a = make_uint4(0, 0, 0, 0), b = a;
            if (r0 + r < rows) {
                const uint4* p = (const uint4*)(w + (size_t)(r0 + r) * cols + k0 + c);
                a = __ldg(p);
                b = __ldg(p + 1);
            }
            *(uint4*)&Ws[r * TS + c] = a;
            *(uint4*)&Ws[r * TS + c + 8] = b;
        }
        load_x(Xs, x, cols, k0, [&](int j) { return t0 + j < T ? t0 + j : -1; });
        __syncthreads();
        mma_tile(Ws, Xs, acc);
        __syncthreads();
    }
    store_y(acc, r0, [&](int r, int c, float v) {
        if (r < rows && t0 + c < T) y[(size_t)(t0 + c) * rows + r] = v;
    });
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
// seq が 0 でなければ、[直前の入力3つ; このバッチの入力 T 個] を seq [3 + T][ch] に残す。投機的デコードで、受け入れた位置までの状態に戻すため
extern "C" __global__ void conv(const float* cur, int cs, const float* w, int dyn, float* hist, float* out, int os, int ch, int cph, int T, float* seq) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    float h0 = hist[c * 3], h1 = hist[c * 3 + 1], h2 = hist[c * 3 + 2];
    if (seq) {
        seq[c] = h0;
        seq[ch + c] = h1;
        seq[2 * ch + c] = h2;
    }
    float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int t = 0; t < T; t++) {
        const float* row = cur + (size_t)t * cs;
        float x = row[c];
        if (seq) seq[(size_t)(3 + t) * ch + c] = x;
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
                                    int ring, half_t* kc, half_t* vc) {
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
        kc[h * HD + t] = f2h(x * clip_scale(x, 22.627416997969522f));
    } else {
        int h = b - 56;
        float x = v[h * HD + t];
        vc[h * HD + t] = f2h(x * rsqrtf(block_sum(x * x) / HD + 1e-6f));
    }
}

// ---- 注意 ----
// 位置 pos のクエリは、global 層なら 0..=pos、SWA 層 (ring > 0) なら直近 EXTENT 個の位置を見る。SWA の KV はリングバッファ (位置 % ring)。
// スコア = q.k / sqrt(HD) + 相対位置バイアス (距離 d < EXTENT のとき r . rel_proj[:, d])。sink はロジット sinks[h]、値はゼロベクトル。

__device__ __forceinline__ float rel_bias(const float* rh, const float* rel_proj, int d) {
    if (d >= EXTENT) return 0.f;
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < NREL; i++) s += rh[i] * rel_proj[i * EXTENT + d];
    return s;
}

__device__ __forceinline__ size_t kv_row(int j, int ring) { return (size_t)(ring ? j % ring : j) * 1024; }

// (T = 1 用) キーの範囲を NSPLIT 個に分けて、部分ごとの (最大値 m、指数和 z、重み付き和 o[HD]) を part に書く。
// グリッドは (ヘッド, NSPLIT) で固定なので、CUDA Graph でも位置に応じて分け方が変わる。128スレッド
#define NSPLIT 16
#define PART (HD + 2)
extern "C" __global__ void attn_part(const float* qx, const half_t* kc, const half_t* vc, const float* rel_proj, float* part, const int* posp, int ring) {
    __shared__ float sc[(MAX_CTX_DEV + NSPLIT - 1) / NSPLIT];
    int pos = *posp, first = ring ? max(0, pos + 1 - EXTENT) : 0, n = pos + 1;
    int h = blockIdx.x, kvh = h / 3, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    int chunk = (n - first + NSPLIT - 1) / NSPLIT, j0 = first + blockIdx.y * chunk, j1 = min(n, j0 + chunk);
    float* out = part + (h * NSPLIT + blockIdx.y) * PART;
    float rh[NREL];
    for (int i = 0; i < NREL; i++) rh[i] = qx[R_OFF + h * NREL + i];
    float4 qv = ((const float4*)(qx + h * HD))[lane];
    for (int j = j0 + warp; j < j1; j += blockDim.x >> 5) {
        float4 kv = h4(kc + kv_row(j, ring) + kvh * HD + lane * 4);
        float s = warp_sum(qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w);
        if (lane == 0) sc[j - j0] = s * rsqrtf((float)HD) + rel_bias(rh, rel_proj, pos - j);
    }
    __syncthreads();
    float m = -INFINITY;
    for (int i = t; i < j1 - j0; i += blockDim.x) m = fmaxf(m, sc[i]);
    m = block_max(m);
    float z = 0.f;
    if (j1 > j0)
        for (int i = t; i < j1 - j0; i += blockDim.x) z += sc[i] = expf(sc[i] - m);
    z = block_sum(z);
    float o = 0.f;
    for (int j = j0; j < j1; j++) o += sc[j - j0] * h2f(vc[kv_row(j, ring) + kvh * HD + t]);
    out[2 + t] = o;
    if (t == 0) {
        out[0] = m;
        out[1] = z;
    }
}

// (T = 1 用) 部分結果と sink をまとめ、出力ゲートをかける。ヘッドごとに1ブロック、128スレッド
extern "C" __global__ void attn_combine(const float* qx, const float* part, float* out) {
    int h = blockIdx.x, t = threadIdx.x;
    const float* p = part + h * NSPLIT * PART;
    float sink = qx[SINK_OFF + h], m = sink;
    for (int s = 0; s < NSPLIT; s++) m = fmaxf(m, p[s * PART]);
    float z = expf(sink - m), o = 0.f;
    for (int s = 0; s < NSPLIT; s++) {
        float f = p[s * PART + 1] > 0.f ? expf(p[s * PART] - m) : 0.f;
        z += p[s * PART + 1] * f;
        o += p[s * PART + 2 + t] * f;
    }
    out[h * HD + t] = o / z * sigmoidf(qx[GATE_OFF + h * HD + t]);
}

// (prefill 用) 32 クエリ x 16 キーのタイルで、K と V を共有メモリに載せて 32 クエリで使い回す (オンライン softmax)。
// グリッドは (ヘッド, T/32)、128スレッド。スレッドはクエリ tid/4 の、次元 (tid%4)*32 から 32 個を受け持つ
#define QT 32
#define KT 16
extern "C" __global__ void attn_tile(const float* qx, const half_t* kc, const half_t* vc, const float* rel_proj, float* out, const int* posp, int ring, int T) {
    __shared__ __align__(16) float Qs[QT][HD];
    __shared__ __align__(16) float Ks[KT][HD + 4];
    __shared__ __align__(16) float Vs[KT][HD];
    __shared__ float Ps[QT][KT];
    __shared__ float Rs[QT][NREL];
    int h = blockIdx.x, kvh = h / 3, tid = threadIdx.x, qi = tid / 4, part = tid % 4;
    int t0 = blockIdx.y * QT, nq = min(QT, T - t0), p0 = *posp + t0;
    for (int i = tid; i < QT * HD; i += blockDim.x) {
        int q = i / HD;
        Qs[q][i % HD] = q < nq ? qx[(size_t)(t0 + q) * QX + h * HD + i % HD] : 0.f;
    }
    for (int i = tid; i < QT * NREL; i += blockDim.x) {
        int q = i / NREL;
        Rs[q][i % NREL] = q < nq ? qx[(size_t)(t0 + q) * QX + R_OFF + h * NREL + i % NREL] : 0.f;
    }
    int pq = p0 + qi; // このスレッドのクエリの位置
    bool valid = qi < nq;
    float m = valid ? qx[(size_t)(t0 + qi) * QX + SINK_OFF + h] : 0.f, z = 1.f; // sink から始める
    float o[32];
    for (int d = 0; d < 32; d++) o[d] = 0.f;
    int pmax = p0 + nq - 1, jlo = ring ? max(0, p0 + 1 - EXTENT) : 0;
    for (int j0 = jlo; j0 <= pmax; j0 += KT) {
        __syncthreads();
        for (int i = tid; i < KT * HD; i += blockDim.x) {
            int k = i / HD, j = j0 + k;
            float kv = 0.f, vv = 0.f;
            if (j <= pmax) {
                kv = h2f(kc[kv_row(j, ring) + kvh * HD + i % HD]);
                vv = h2f(vc[kv_row(j, ring) + kvh * HD + i % HD]);
            }
            Ks[k][i % HD] = kv;
            Vs[k][i % HD] = vv;
        }
        __syncthreads();
        // このスレッドはキー part*4 .. part*4+3 のスコアを計算する
        float s[4], tmax = -INFINITY;
        for (int kk = 0; kk < 4; kk++) {
            int k = part * 4 + kk, j = j0 + k, d = pq - j;
            bool ok = valid && j <= pq && (!ring || d < EXTENT);
            float acc = 0.f;
            if (ok) {
                for (int e = 0; e < HD; e += 4) {
                    float4 a = *(const float4*)&Qs[qi][e], b = *(const float4*)&Ks[k][e];
                    acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
                }
                acc = acc * rsqrtf((float)HD) + rel_bias(Rs[qi], rel_proj, d);
            }
            s[kk] = ok ? acc : -INFINITY;
            tmax = fmaxf(tmax, s[kk]);
        }
        // 同じクエリの4スレッドは隣り合うレーン
        tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 1));
        tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 2));
        float mn = fmaxf(m, tmax), f = expf(m - mn), ps = 0.f;
        for (int kk = 0; kk < 4; kk++) {
            float p = expf(s[kk] - mn);
            Ps[qi][part * 4 + kk] = p;
            ps += p;
        }
        ps += __shfl_xor_sync(0xffffffffu, ps, 1);
        ps += __shfl_xor_sync(0xffffffffu, ps, 2);
        z = z * f + ps;
        m = mn;
        __syncwarp();
        for (int d = 0; d < 32; d++) o[d] *= f;
        for (int k = 0; k < KT; k++) {
            float p = Ps[qi][k];
            const float* vr = &Vs[k][part * 32];
            for (int d = 0; d < 32; d++) o[d] += p * vr[d];
        }
    }
    if (valid) {
        const float* gate = qx + (size_t)(t0 + qi) * QX + GATE_OFF + h * HD + part * 32;
        float* dst = out + (size_t)(t0 + qi) * 3072 + h * HD + part * 32;
        for (int d = 0; d < 32; d++) dst[d] = o[d] / z * sigmoidf(gate[d]);
    }
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
// Y[a][r] = amax/2688 * sum_c W_e[r][c] X[a / xdiv][c]。グリッドは (rows/64, expert)、128スレッド
// e2m1 x e4m3 は BF16 で誤差なく表せるので、amax/2688 は最後に掛ける
extern "C" __global__ void gemm_nvfp4_grouped(const u8* __restrict__ w, const u8* __restrict__ s, const float* amax, int rows, int cols,
                                              const float* __restrict__ x, int xdiv, const int* order, const int* off, float* y) {
    __shared__ __align__(16) unsigned short Ws[64 * TS], Xs[2 * 32 * TS];
    __shared__ int as[32];
    int e = blockIdx.y, beg = off[e], end = off[e + 1];
    if (beg == end) return;
    int tid = threadIdx.x, r0 = blockIdx.x * 64;
    const u8* we = w + (size_t)e * rows * (cols / 2);
    const u8* se = s + (size_t)e * rows * (cols / 16);
    float g = amax[e] / 2688.f;
    for (int c0 = beg; c0 < end; c0 += 32) {
        if (tid < 32) as[tid] = c0 + tid < end ? order[c0 + tid] : -1;
        float acc[4][4] = {};
        for (int k0 = 0; k0 < cols; k0 += 32) {
            __syncthreads();
            // W のタイル: 64行 x 32列。1スレッドが1行の 16 要素 (8 バイトとスケール1個) を展開する
            {
                int r = tid / 2, c = (tid % 2) * 16;
                uint2 pk = *(const uint2*)(we + (size_t)(r0 + r) * (cols / 2) + (k0 + c) / 2);
                float sc = e4m3(se[(size_t)(r0 + r) * (cols / 16) + (k0 + c) / 16]);
                unsigned v[8];
                for (int i = 0; i < 4; i++) {
                    v[i] = pack2(e2m1(pk.x >> (8 * i)) * sc, e2m1(pk.x >> (8 * i + 4)) * sc);
                    v[i + 4] = pack2(e2m1(pk.y >> (8 * i)) * sc, e2m1(pk.y >> (8 * i + 4)) * sc);
                }
                *(uint4*)&Ws[r * TS + c] = make_uint4(v[0], v[1], v[2], v[3]);
                *(uint4*)&Ws[r * TS + c + 8] = make_uint4(v[4], v[5], v[6], v[7]);
            }
            load_x(Xs, x, cols, k0, [&](int j) { return as[j] >= 0 ? as[j] / xdiv : -1; });
            __syncthreads();
            mma_tile(Ws, Xs, acc);
        }
        store_y(acc, r0, [&](int r, int c, float v) {
            if (as[c] >= 0) y[(size_t)as[c] * rows + r] = v * g;
        });
        __syncthreads();
    }
}

// (prefill 用) MXFP8 の expert を、PCIe で送った置き場から読んで掛ける。置き場には expert e0 から gridDim.y 個が並ぶ
// Y[a][r] = sum_c W_e[r][c] X[a / xdiv][c]。グリッドは (rows/64, expert 数)、128スレッド
// E4M3 x 2^(s-127) は BF16 で誤差なく表せる
extern "C" __global__ void gemm_mxfp8_grouped(const u8* __restrict__ w, const u8* __restrict__ s, int rows, int cols, const float* __restrict__ x, int xdiv,
                                              const int* order, const int* off, int e0, float* y) {
    __shared__ __align__(16) unsigned short Ws[64 * TS], Xs[2 * 32 * TS];
    __shared__ int as[32];
    int e = e0 + blockIdx.y, beg = off[e], end = off[e + 1];
    if (beg == end) return;
    int tid = threadIdx.x, r0 = blockIdx.x * 64;
    const u8* we = w + (size_t)blockIdx.y * rows * cols;
    const u8* se = s + (size_t)blockIdx.y * rows * (cols / 32);
    for (int c0 = beg; c0 < end; c0 += 32) {
        if (tid < 32) as[tid] = c0 + tid < end ? order[c0 + tid] : -1;
        float acc[4][4] = {};
        for (int k0 = 0; k0 < cols; k0 += 32) {
            __syncthreads();
            {
                int r = tid / 2, c = (tid % 2) * 16;
                uint4 pk = *(const uint4*)(we + (size_t)(r0 + r) * cols + k0 + c);
                float sc = __uint_as_float((unsigned)se[(size_t)(r0 + r) * (cols / 32) + (k0 + c) / 32] << 23);
                unsigned u[4] = {pk.x, pk.y, pk.z, pk.w}, v[8];
                for (int i = 0; i < 4; i++) {
                    v[2 * i] = pack2(e4m3(u[i]) * sc, e4m3(u[i] >> 8) * sc);
                    v[2 * i + 1] = pack2(e4m3(u[i] >> 16) * sc, e4m3(u[i] >> 24) * sc);
                }
                *(uint4*)&Ws[r * TS + c] = make_uint4(v[0], v[1], v[2], v[3]);
                *(uint4*)&Ws[r * TS + c + 8] = make_uint4(v[4], v[5], v[6], v[7]);
            }
            load_x(Xs, x, cols, k0, [&](int j) { return as[j] >= 0 ? as[j] / xdiv : -1; });
            __syncthreads();
            mma_tile(Ws, Xs, acc);
        }
        store_y(acc, r0, [&](int r, int c, float v) {
            if (as[c] >= 0) y[(size_t)as[c] * rows + r] = v;
        });
        __syncthreads();
    }
}

// (prefill 用) order[base..base+n] の割り当てだけ SITU をかける
extern "C" __global__ void situ_sel(const float* gu, float* act, const int* order, int base, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * FF) return;
    int a = order[base + i / FF], j = i % FF;
    act[(size_t)a * FF + j] = situ(gu[(size_t)a * 2 * FF + 2 * j], gu[(size_t)a * 2 * FF + 2 * j + 1]);
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

// ---- 投機的デコードの検証 ----

// 畳み込みの状態を、検証したバッチのうち先頭 a 個の入力を受け入れた時点に戻す。
// tab[3e .. 3e+2] = (seq, hist, ch)。新しい状態は seq の行 a, a+1, a+2。グリッドは (ch/256, 項目の数)
extern "C" __global__ void conv_rewind(const unsigned long long* tab, int a) {
    const float* seq = (const float*)tab[3 * blockIdx.y];
    float* hist = (float*)tab[3 * blockIdx.y + 1];
    int ch = (int)tab[3 * blockIdx.y + 2], c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    for (int k = 0; k < 3; k++) hist[c * 3 + k] = seq[(size_t)(a + k) * ch + c];
}

// 行ごとの、先頭 n 個のうち最大の値の位置 (同じ値なら小さい位置)。行は stride 個おき。ブロックは行ごと、1024スレッド
extern "C" __global__ void argmax_rows(const float* x, int stride, int n, int* out) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    const float* r = x + (size_t)blockIdx.x * stride;
    float v = -INFINITY;
    int i = 0;
    for (int j = threadIdx.x; j < n; j += blockDim.x)
        if (r[j] > v) v = r[j], i = j;
    for (int o = 16; o > 0; o >>= 1) {
        float v2 = __shfl_xor_sync(0xffffffffu, v, o);
        int i2 = __shfl_xor_sync(0xffffffffu, i, o);
        if (v2 > v || (v2 == v && i2 < i)) v = v2, i = i2;
    }
    if ((threadIdx.x & 31) == 0) bv[threadIdx.x >> 5] = v, bi[threadIdx.x >> 5] = i;
    __syncthreads();
    if (threadIdx.x < 32) {
        v = threadIdx.x < (blockDim.x >> 5) ? bv[threadIdx.x] : -INFINITY;
        i = threadIdx.x < (blockDim.x >> 5) ? bi[threadIdx.x] : 0;
        for (int o = 16; o > 0; o >>= 1) {
            float v2 = __shfl_xor_sync(0xffffffffu, v, o);
            int i2 = __shfl_xor_sync(0xffffffffu, i, o);
            if (v2 > v || (v2 == v && i2 < i)) v = v2, i = i2;
        }
        if (threadIdx.x == 0) out[blockIdx.x] = i;
    }
}
