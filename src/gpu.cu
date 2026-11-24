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
// PDL (Programmatic Dependent Launch)。前のカーネルが終わる (書いたものが見える) まで待ち、すぐ次のカーネルの起動を許す。
// 次のカーネルのブロックは、このカーネルの最後の波と重なって SM に載り、ここで待つ。起動の待ちが隠れる
#define PDL() asm volatile("griddepcontrol.wait;\n\tgriddepcontrol.launch_dependents;" ::: "memory")

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
    // 重みは前のカーネルに依存しないので、待つあいだに L2 へ読み込んでおく (128 バイトずつ)
    for (int i = lane * 8; i < cols / 8; i += 256) asm volatile("prefetch.global.L2 [%0];" :: "l"(wr + i));
    PDL();
    float s = 0.f;
    #pragma unroll 4
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
#define WST 72 // f16 の行列積の共有メモリの行の長さ (64 + 8 の詰め物)

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

__device__ __forceinline__ unsigned h2pack(float lo, float hi) {
    unsigned d;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
    return d;
}

// 共有メモリの 8x8 の行列 4つを読む。レーン l は行列 l/8 の行 l%8 の場所を渡す
__device__ __forceinline__ void ldsm4(const void* p, unsigned r[4]) {
    unsigned addr = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

// Xs [列][K] (行の長さ st) から、列 n0..n0+15、K k0..k0+15 の B フラグメントを読む。b[0], b[1] が列 n0..n0+7、b[2], b[3] が n0+8..n0+15
__device__ __forceinline__ void ldb(const unsigned* Xs, int st, int n0, int k0, unsigned b[4]) {
    int l = threadIdx.x & 31;
    ldsm4((const half_t*)Xs + (n0 + (l & 7) + (l >> 4) * 8) * st + k0 + ((l >> 3) & 1) * 8, b);
}

__device__ __forceinline__ void mma_f16(float c[4], const unsigned a[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
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

// Y[t][r] = sum_c W[r][c] X[t][c]。W は BF16 の [rows, cols]、cols は 64 の倍数。グリッドは (rows/64, T/32)、128スレッド。
// 重みと活性を f16 にしてテンソルコア (f32 累積) で掛ける。BF16 の重みは f16 の正規数の範囲なら誤差なく移る。
// ブロックは 64行 x 32列 (トークン) を受け持ち、K は 64 ずつ共有メモリに載せる。次の K の分は、今の分の積和の前にレジスタへ読んでおく
extern "C" __global__ void gemm_bf16(const bf16* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, int rows, int cols, int T) {
    PDL();
    __shared__ __align__(16) unsigned Ws[64 * WST / 2], Xs[2][32 * WST / 2];
    int r0 = blockIdx.x * 64, t0 = blockIdx.y * 32, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    // 重みはスレッドあたり 行 tid/2 の 32 要素、活性は トークン tid/4 の 16 要素
    int wr = tid / 2, wc = (tid % 2) * 32, xj = tid / 4, xc = (tid % 4) * 16;
    bool wok = r0 + wr < rows, xok = t0 + xj < T;
    uint4 wv[4];
    float4 xv[4];
    auto fetch = [&](int k0) {
        #pragma unroll
        for (int q = 0; q < 4; q++) {
            wv[q] = wok ? __ldg((const uint4*)(w + (size_t)(r0 + wr) * cols + k0 + wc) + q) : make_uint4(0, 0, 0, 0);
            xv[q] = xok ? ((const float4*)(x + (size_t)(t0 + xj) * cols + k0 + xc))[q] : make_float4(0.f, 0.f, 0.f, 0.f);
        }
    };
    auto bf2h = [](unsigned u) { return h2pack(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u)); };
    float acc[4][4] = {};
    fetch(0);
    for (int k0 = 0; k0 < cols; k0 += 64) {
        __syncthreads();
        #pragma unroll
        for (int q = 0; q < 4; q++) *(uint4*)(Ws + wr * (WST / 2) + wc / 2 + q * 4) = make_uint4(bf2h(wv[q].x), bf2h(wv[q].y), bf2h(wv[q].z), bf2h(wv[q].w));
        #pragma unroll
        for (int q = 0; q < 4; q += 2)
            {
                const float* f = (const float*)&xv[q];
                unsigned hi[4], lo[4];
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    hi[i] = h2pack(f[2 * i], f[2 * i + 1]);
                    lo[i] = h2pack(f[2 * i] - h2f(hi[i] & 0xffff), f[2 * i + 1] - h2f(hi[i] >> 16));
                }
                *(uint4*)(Xs[0] + xj * (WST / 2) + xc / 2 + q * 2) = make_uint4(hi[0], hi[1], hi[2], hi[3]);
                *(uint4*)(Xs[1] + xj * (WST / 2) + xc / 2 + q * 2) = make_uint4(lo[0], lo[1], lo[2], lo[3]);
            }
        __syncthreads();
        if (k0 + 64 < cols) fetch(k0 + 64);
        #pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            int r = warp * 16 + g, k = ks * 8 + t;
            unsigned a[4] = {Ws[r * (WST / 2) + k], Ws[(r + 8) * (WST / 2) + k], Ws[r * (WST / 2) + k + 4], Ws[(r + 8) * (WST / 2) + k + 4]};
            #pragma unroll
            for (int j = 0; j < 4; j++)
                #pragma unroll
                for (int h = 0; h < 2; h++) mma_f16(acc[j], a, Xs[h][(j * 8 + g) * (WST / 2) + k], Xs[h][(j * 8 + g) * (WST / 2) + k + 4]);
        }
    }
    store_y(acc, r0, [&](int r, int c, float v) {
        if (r < rows && t0 + c < T) y[(size_t)(t0 + c) * rows + r] = v;
    });
}

// ---- Q8 の行列 (SQ 形式の本体): [rows] の f32 スケールのあとに int8 [rows][cols] ----
// int8 4つを u ^ 0x80808080 にしておくと、バイト k を f32 の仮数部に置いた値は 2^23 + 128 + q になる
__device__ __forceinline__ float q8f(unsigned u, int k) { return __int_as_float(__byte_perm(u, 0x4B000000u, 0x7540 | k)) - 8388736.f; }

// int8 16個 (v) と x[0..16] の内積
__device__ __forceinline__ float q8dot16(uint4 v, const float4* xx) {
    unsigned w[4] = {v.x ^ 0x80808080u, v.y ^ 0x80808080u, v.z ^ 0x80808080u, v.w ^ 0x80808080u};
    float s = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        float4 a = xx[j];
        s += q8f(w[j], 0) * a.x + q8f(w[j], 1) * a.y + q8f(w[j], 2) * a.z + q8f(w[j], 3) * a.w;
    }
    return s;
}

__device__ __forceinline__ const uint4* q8row(const float* w, int rows, int cols, int row) { return (const uint4*)((const u8*)(w + rows) + (size_t)row * cols); }

// y = W x (T = 1 用)。W は Q8、cols は 16 の倍数。1行を1ワープで計算する
extern "C" __global__ void mv_q8(const float* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, int rows, int cols) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= rows) return;
    const uint4* wr = q8row(w, rows, cols, row);
    for (int i = lane * 8; i < cols / 16; i += 256) asm volatile("prefetch.global.L2 [%0];" :: "l"(wr + i));
    PDL();
    float s = 0.f;
    #pragma unroll 4
    for (int i = lane; i < cols / 16; i += 32) s += q8dot16(__ldg(wr + i), (const float4*)(x + i * 16));
    s = warp_sum(s);
    if (lane == 0) y[row] = s * w[row];
}

// (T = 1 用) mv3_bf16 の Q8 版
extern "C" __global__ void mv3_q8(const float* w0, const float* x0, float* y0, int r0, const float* w1, const float* x1, float* y1, int r1,
                                  const float* w2, const float* x2, float* y2, int r2) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= r0 + r1 + r2) return;
    const float* w = w0;
    const float* x = x0;
    float* y = y0;
    int rows = r0;
    if (row >= r0 + r1) w = w2, x = x2, y = y2, row -= r0 + r1, rows = r2;
    else if (row >= r0) w = w1, x = x1, y = y1, row -= r0, rows = r1;
    const uint4* wr = q8row(w, rows, D, row);
    for (int i = lane * 8; i < D / 16; i += 256) asm volatile("prefetch.global.L2 [%0];" :: "l"(wr + i));
    PDL();
    float s = 0.f;
    #pragma unroll 4
    for (int i = lane; i < D / 16; i += 32) s += q8dot16(__ldg(wr + i), (const float4*)(x + i * 16));
    s = warp_sum(s);
    if (lane == 0) y[row] = s * w[row];
}

// (T <= 8 用) mv_bf16_t の Q8 版。1ワープが 4 行を受け持ち、活性の読み込みを 4 行で使い回す (rows は 4 の倍数)
extern "C" __global__ void mv_q8_t(const float* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, int rows, int cols, int T) {
    int row0 = ((blockIdx.x * blockDim.x + threadIdx.x) >> 5) * 4, lane = threadIdx.x & 31;
    if (row0 >= rows) return;
    const uint4* wr[4];
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        wr[r] = q8row(w, rows, cols, row0 + r);
        for (int i = lane * 8; i < cols / 16; i += 256) asm volatile("prefetch.global.L2 [%0];" :: "l"(wr[r] + i));
    }
    PDL();
    float acc[4][8] = {};
    for (int i = lane; i < cols / 16; i += 32) {
        float wf[4][16];
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            uint4 v = __ldg(wr[r] + i);
            unsigned u[4] = {v.x ^ 0x80808080u, v.y ^ 0x80808080u, v.z ^ 0x80808080u, v.w ^ 0x80808080u};
            #pragma unroll
            for (int j = 0; j < 16; j++) wf[r][j] = q8f(u[j / 4], j % 4);
        }
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            if (t >= T) break;
            const float4* xx = (const float4*)(x + (size_t)t * cols + i * 16);
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float4 a = xx[j];
                #pragma unroll
                for (int r = 0; r < 4; r++) acc[r][t] += wf[r][4 * j] * a.x + wf[r][4 * j + 1] * a.y + wf[r][4 * j + 2] * a.z + wf[r][4 * j + 3] * a.w;
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        float sc = w[row0 + r];
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            if (t >= T) break;
            float s = warp_sum(acc[r][t]);
            if (lane == 0) y[(size_t)t * rows + row0 + r] = s * sc;
        }
    }
}

// logits: y[t][r] = sum_c W[r][c] x[t][c] (t < T <= 8, c < D)。W は S6_K を並べ直したもの (Pack::s6): 下位4ビット [rows][D/2]、上位2ビット [rows][D/4]、
// 64 要素ごとのスケール [rows][D/64] (f32)。値は (q - 31.5) * スケール。1ワープが 4 行を受け持ち (活性の読み込みを使い回す)、レーン l は 256 要素ごとの 8l..8l+7 を受け持つ
extern "C" __global__ void mv_s6(const u8* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, int rows, int T) {
    int row0 = ((blockIdx.x * blockDim.x + threadIdx.x) >> 5) * 4, lane = threadIdx.x & 31;
    if (row0 >= rows) return;
    const unsigned* lo = (const unsigned*)(w + (size_t)row0 * (D / 2));
    const unsigned short* hi = (const unsigned short*)(w + (size_t)rows * (D / 2) + (size_t)row0 * (D / 4));
    const float* sc = (const float*)(w + (size_t)rows * (D / 2 + D / 4)) + (size_t)row0 * (D / 64);
    // 重みは前のカーネルに依存しないので、待つあいだに L2 へ読み込んでおく (4 行の下位 4608 バイト、上位 2304 バイト)
    if (lane < 36) asm volatile("prefetch.global.L2 [%0];" :: "l"(lo + lane * 32));
    if (lane < 18) asm volatile("prefetch.global.L2 [%0];" :: "l"(hi + lane * 64));
    PDL();
    float acc[4][8] = {};
    for (int s = 0; s < D / 256; s++) {
        float q[4][8], g[4];
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            unsigned l = __ldg(lo + r * (D / 8) + s * 32 + lane), h = __ldg(hi + r * (D / 8) + s * 32 + lane);
            g[r] = __ldg(sc + r * (D / 64) + s * 4 + lane / 8);
            #pragma unroll
            for (int e = 0; e < 8; e++) q[r][e] = (float)(((l >> (4 * e)) & 15u) | (((h >> (2 * e)) & 3u) << 4)) - 31.5f;
        }
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            if (t >= T) break;
            const float4* xx = (const float4*)(x + (size_t)t * D + s * 256 + lane * 8);
            float4 a = xx[0], b = xx[1];
            #pragma unroll
            for (int r = 0; r < 4; r++)
                acc[r][t] += g[r] * (q[r][0] * a.x + q[r][1] * a.y + q[r][2] * a.z + q[r][3] * a.w + q[r][4] * b.x + q[r][5] * b.y + q[r][6] * b.z + q[r][7] * b.w);
        }
    }
    #pragma unroll
    for (int r = 0; r < 4; r++)
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            if (t >= T) break;
            float s = warp_sum(acc[r][t]);
            if (lane == 0) y[(size_t)t * rows + row0 + r] = s;
        }
}
// ---- (prefill 用) int8 のテンソルコアで計算する Q8 の行列積 ----
// 活性を 64 要素ごとに int8 に量子化する (Strata と同じ精度)。グリッドは (n, cols / 64 / 8)、256 スレッド (ワープが 64 要素の組 1つ)
extern "C" __global__ void quant_q8(const float* x, int cols, signed char* q, float* s) {
    PDL();
    int r = blockIdx.x, gi = blockIdx.y * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31, G = cols / 64;
    if (gi >= G) return;
    float2 v = ((const float2*)(x + (size_t)r * cols + gi * 64))[lane];
    float mx = fmaxf(fabsf(v.x), fabsf(v.y));
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    float sc = mx > 0.f ? mx / 127.f : 1.f;
    char2 c = make_char2((signed char)__float2int_rn(v.x / sc), (signed char)__float2int_rn(v.y / sc));
    ((char2*)(q + (size_t)r * cols + gi * 64))[lane] = c;
    if (lane == 0) s[(size_t)r * G + gi] = sc;
}

__device__ __forceinline__ void mma_s8(int c[4], const unsigned a[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Y[t][r] = sc_w[r] * sum_g sc_x[t][g] * sum_{k in g} q_w[r][k] q_x[t][k]。W は Q8、x は quant_q8 で量子化したもの。cols は 64 の倍数。
// グリッドは (rows/64, T/32)、128スレッド。ブロックは 64行 x 32列 (トークン)、K は 64 (活性の量子化の単位) ずつ共有メモリに載せ、m16n8k32 の int8 mma 2 回で積和する
#define Q8ST 20 // 共有メモリの行の長さ (u32)。64 バイト + 16 の詰め物
extern "C" __global__ void gemm_q8i(const float* __restrict__ w, const signed char* __restrict__ xq, const float* __restrict__ xs, float* __restrict__ y, int rows, int cols, int T) {
    PDL();
    __shared__ __align__(16) unsigned Ws[64 * Q8ST], Xs[32 * Q8ST];
    __shared__ float sxs[32];
    int r0 = blockIdx.x * 64, t0 = blockIdx.y * 32, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3, G = cols / 64;
    // 重みはスレッドあたり 行 tid/2 の 32 バイト、活性は トークン tid/4 の 16 バイト
    int wr = tid / 2, wc = (tid % 2) * 32, xj = tid / 4, xc = (tid % 4) * 16;
    bool wok = r0 + wr < rows, xok = t0 + xj < T;
    const u8* wq = (const u8*)(w + rows);
    uint4 wv[2], xv;
    auto fetch = [&](int k0) {
        #pragma unroll
        for (int q = 0; q < 2; q++) wv[q] = wok ? __ldg((const uint4*)(wq + (size_t)(r0 + wr) * cols + k0 + wc) + q) : make_uint4(0, 0, 0, 0);
        xv = xok ? *(const uint4*)(xq + (size_t)(t0 + xj) * cols + k0 + xc) : make_uint4(0, 0, 0, 0);
    };
    float acc[4][4] = {};
    fetch(0);
    for (int k0 = 0; k0 < cols; k0 += 64) {
        __syncthreads();
        *(uint4*)(Ws + wr * Q8ST + wc / 4) = wv[0];
        *(uint4*)(Ws + wr * Q8ST + wc / 4 + 4) = wv[1];
        *(uint4*)(Xs + xj * Q8ST + xc / 4) = xv;
        if (tid < 32) sxs[tid] = t0 + tid < T ? xs[(size_t)(t0 + tid) * G + k0 / 64] : 0.f;
        __syncthreads();
        if (k0 + 64 < cols) fetch(k0 + 64);
        int seg[4][4] = {};
        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            int r = warp * 16 + g, k = kk * 8 + t;
            unsigned a[4] = {Ws[r * Q8ST + k], Ws[(r + 8) * Q8ST + k], Ws[r * Q8ST + k + 4], Ws[(r + 8) * Q8ST + k + 4]};
            #pragma unroll
            for (int j = 0; j < 4; j++) mma_s8(seg[j], a, Xs[(j * 8 + g) * Q8ST + k], Xs[(j * 8 + g) * Q8ST + k + 4]);
        }
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float x0 = sxs[j * 8 + 2 * t], x1 = sxs[j * 8 + 2 * t + 1];
            acc[j][0] += (float)seg[j][0] * x0, acc[j][1] += (float)seg[j][1] * x1;
            acc[j][2] += (float)seg[j][2] * x0, acc[j][3] += (float)seg[j][3] * x1;
        }
    }
    store_y(acc, r0, [&](int r, int c, float v) {
        if (r < rows && t0 + c < T) y[(size_t)(t0 + c) * rows + r] = v * w[r];
    });
}

// 長さ D のベクトル gridDim.x 本をそれぞれ RMS 正規化する。並びは [種類][T][D] で、重みは種類ごとに wstride ずつずれる (0 なら共通)
extern "C" __global__ void rmsnorm(const float* x, const float* w, float* y, int wstride, int T) {
    PDL();
    x += (size_t)blockIdx.x * D;
    y += (size_t)blockIdx.x * D;
    if (w) w += (blockIdx.x / T) * wstride;
    // 256 スレッド。値を先にまとめて読む
    float v[D / 256], ss = 0.f;
    #pragma unroll
    for (int j = 0; j < D / 256; j++) v[j] = x[threadIdx.x + j * 256];
    #pragma unroll
    for (int j = 0; j < D / 256; j++) ss += v[j] * v[j];
    float k = rsqrtf(block_sum(ss) / D + 1e-6f);
    #pragma unroll
    for (int j = 0; j < D / 256; j++) y[threadIdx.x + j * 256] = v[j] * k * (w ? w[threadIdx.x + j * 256] : 1.f);
}

// 深さ方向の注意。状態は blocks[0..nb] と cur (nb + 1 <= MIX_NS)。ブロックは (query, トークン) で 256 スレッド。out は [query][T][D]
// nw が 0 でなければ、続けて rmsnorm (重みは query ごとに nws ずつずれる) をかける。
// 各スレッドは要素 threadIdx.x + 256j を受け持ち、全状態の値を先にまとめて読む (読み込みを1つずつ待つと L2 の待ちが積み重なる)。
// 足す順は、状態ごとに block_sum で足していたときと同じ
#define MIX_NS 5
#define MIX_TH 768 // スレッド数 (D の約数で 32 の倍数)
extern "C" __global__ void mix(const float* blocks, int nb, const float* cur, const float* queries, float* out, int T, const float* nw, int nws, float* rms) {
    PDL();
    int t = blockIdx.y, ns = nb + 1;
    const float* q = queries + blockIdx.x * D;
    float v[D / MIX_TH][MIX_NS], qv[D / MIX_TH];
    #pragma unroll
    for (int j = 0; j < D / MIX_TH; j++) {
        int i = threadIdx.x + j * MIX_TH;
        qv[j] = q[i];
        #pragma unroll
        for (int s = 0; s < MIX_NS; s++)
            if (s < ns) v[j][s] = (s < nb ? blocks + ((size_t)s * T + t) * D : cur + (size_t)t * D)[i];
    }
    float ss[MIX_NS] = {}, dq[MIX_NS] = {}, sc[MIX_NS], inv[MIX_NS];
    #pragma unroll
    for (int j = 0; j < D / MIX_TH; j++) {
        #pragma unroll
        for (int s = 0; s < MIX_NS; s++)
            if (s < ns) {
                ss[s] += v[j][s] * v[j][s];
                dq[s] += v[j][s] * qv[j];
            }
    }
    __shared__ float red[MIX_TH / 32][2 * MIX_NS];
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    #pragma unroll
    for (int s = 0; s < MIX_NS; s++) {
        float a = warp_sum(ss[s]), b = warp_sum(dq[s]);
        if (lane == 0) red[w][s] = a, red[w][MIX_NS + s] = b;
    }
    __syncthreads();
    #pragma unroll
    for (int s = 0; s < MIX_NS; s++) {
        float a = warp_sum(lane < MIX_TH / 32 ? red[lane][s] : 0.f), b = warp_sum(lane < MIX_TH / 32 ? red[lane][MIX_NS + s] : 0.f);
        inv[s] = rsqrtf(a / D + 1e-6f);
        sc[s] = b * inv[s] * rsqrtf((float)D);
    }
    float m = -INFINITY, z = 0.f;
    #pragma unroll
    for (int s = 0; s < MIX_NS; s++)
        if (s < ns) m = fmaxf(m, sc[s]);
    #pragma unroll
    for (int s = 0; s < MIX_NS; s++)
        if (s < ns) z += sc[s] = expf(sc[s] - m);
    float o[D / MIX_TH], oss = 0.f;
    #pragma unroll
    for (int j = 0; j < D / MIX_TH; j++) {
        o[j] = 0.f;
        #pragma unroll
        for (int s = 0; s < MIX_NS; s++)
            if (s < ns) o[j] += sc[s] / z * inv[s] * v[j][s];
        oss += o[j] * o[j];
    }
    // rms が 0 でなければ、出力の RMS の逆数を rms[query][T] に書く (control が使う)
    float k = nw || rms ? rsqrtf(block_sum(oss) / D + 1e-6f) : 1.f;
    if (rms && threadIdx.x == 0) rms[blockIdx.x * T + t] = k;
    if (nw) nw += blockIdx.x * nws;
    #pragma unroll
    for (int j = 0; j < D / MIX_TH; j++) {
        int i = threadIdx.x + j * MIX_TH;
        out[((size_t)blockIdx.x * T + t) * D + i] = nw ? o[j] * k * nw[i] : o[j];
    }
}

// control の前半: トークンごとに [rmsnorm(cur), rmsnorm(msg)] を out [T][2D] に並べる。ブロックは (2, T)
extern "C" __global__ void norm2(const float* cur, const float* msg, float* out) {
    PDL();
    int t = blockIdx.y;
    const float* x = (blockIdx.x ? msg : cur) + (size_t)t * D;
    float ss = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) ss += x[i] * x[i];
    float k = rsqrtf(block_sum(ss) / D + 1e-6f);
    for (int i = threadIdx.x; i < D; i += blockDim.x) out[(size_t)t * 2 * D + blockIdx.x * D + i] = x[i] * k;
}

// control の後半: gf は [T][gain D, forget D]。cur を上書きしてよい。ブロックは (D/256, T)
extern "C" __global__ void control(float* cur, const float* msg, const float* gf) {
    PDL();
    int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= D) return;
    size_t j = (size_t)t * D + i;
    const float* g = gf + (size_t)t * 2 * D;
    float c = cur[j];
    cur[j] = c - tanhf(g[D + i]) * c + tanhf(g[i]) * msg[j];
}

// 因果的な短い畳み込みと残差。w は [ch][4] で k=3 が現在、hist は [ch][3]。グリッドは (ch/256, T/CONV_T) で、ブロックは CONV_T トークンを順に流す
// 入力の行は cs 要素おき、出力の行は os 要素おき。dyn >= 0 なら入力の行の dyn 番目からがヘッドごとの [lag] のタップ
// seq が 0 でなければ、[直前の入力3つ; このバッチの入力 T 個] を seq [3 + T][ch] に残す。投機的デコードで、受け入れた位置までの状態に戻すため
// res が 0 でなければ、出力を res にも足す (残差)。ブロックは前のブロックの入力を読むので、T > CONV_T なら out と cur は別の場所にする
#define CONV_T 16
extern "C" __global__ void conv(const float* cur, int cs, const float* w, int dyn, float* hist, float* out, int os, int ch, int cph, int T, float* seq, float* res) {
    PDL();
    int c = blockIdx.x * blockDim.x + threadIdx.x, t0 = blockIdx.y * CONV_T, t1 = min(T, t0 + CONV_T);
    if (c >= ch) return;
    auto in = [&](int t) { return t < 0 ? hist[c * 3 + 3 + t] : cur[(size_t)t * cs + c]; };
    float h0 = in(t0 - 3), h1 = in(t0 - 2), h2 = in(t0 - 1);
    if (seq && t0 == 0) {
        seq[c] = h0;
        seq[ch + c] = h1;
        seq[2 * ch + c] = h2;
    }
    float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int t = t0; t < t1; t++) {
        const float* row = cur + (size_t)t * cs;
        float x = row[c];
        if (seq) seq[(size_t)(3 + t) * ch + c] = x;
        float o = w0 * h0 + w1 * h1 + w2 * h2 + w3 * x;
        if (dyn >= 0) {
            const float* d = row + dyn + (c / cph) * 4;
            o += d[0] * x + d[1] * h2 + d[2] * h1 + d[3] * h0;
        }
        float y = (x + o) * (dyn >= 0 ? 1.f / 3.f : rsqrtf(5.f));
        out[(size_t)t * os + c] = y;
        if (res) res[(size_t)t * os + c] += y;
        h0 = h1;
        h1 = h2;
        h2 = x;
    }
    // hist を読むのは先頭のブロックだけなので、書き換えも先頭のブロックが最後に行う
    if (t0 == 0) {
        if (T > CONV_T) h0 = in(T - 3), h1 = in(T - 2), h2 = in(T - 1);
        hist[c * 3] = h0;
        hist[c * 3 + 1] = h1;
        hist[c * 3 + 2] = h2;
    }
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
    PDL();
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

// (T <= 8 用) キーの範囲を NSPLIT 個に分けて、部分ごとの (最大値 m、指数和 z、重み付き和 o[HD]) をクエリのヘッドごとに part に書く。
// GQA で KV ヘッドを共有する 3 つのクエリのヘッドを 1 ブロックでまとめて計算し、KV を一度だけ読む。
// グリッドは (KV ヘッド 8, NSPLIT * T) で固定なので、CUDA Graph でも位置に応じて分け方が変わる。128 スレッド (4 ワープ)
#define NSPLIT 32
#define PART (HD + 2)
// 分割の中は SUB 個ずつスコアを共有メモリに置き、オンライン softmax でまとめる (文脈が長くても共有メモリは一定)。
// kc, vc は VRAM でも、GPU から直接読める主記憶 (長い文脈の global 層) でもよい
#define SUB 1024
__device__ void attn_combine(const float* qx, const float* part, float* out, int h, int tt);

// 最後に終わった分割のブロックが、その 3 ヘッドの部分結果を合わせて ob [T][3072] に書く。done は KV ヘッドごとの終わった分割の数 [T][24] (最後のブロックが 0 に戻す)
extern "C" __global__ void attn_part(const float* qx, const half_t* kc, const half_t* vc, const float* rel_proj, float* part, const int* posp, int ring,
                                     float* ob, unsigned* done) {
    PDL();
    __shared__ float sc[3][SUB];
    __shared__ float osum[4][3][HD];
    int tt = blockIdx.y / NSPLIT, split = blockIdx.y % NSPLIT, kvh = blockIdx.x;
    const float* q0 = qx + (size_t)tt * QX;
    int pos = *posp + tt, first = ring ? max(0, pos + 1 - EXTENT) : 0, n = pos + 1;
    int chunk = (n - first + NSPLIT - 1) / NSPLIT, j0 = first + split * chunk, j1 = min(n, j0 + chunk);
    int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    // レーンは次元 4 * lane..+3 と、相対位置の成分 lane (< NREL) を持つ
    float4 q[3];
    float rl[3];
    #pragma unroll
    for (int h = 0; h < 3; h++) {
        int head = kvh * 3 + h;
        q[h] = ((const float4*)(q0 + head * HD))[lane];
        rl[h] = lane < NREL ? q0[R_OFF + head * NREL + lane] : 0.f;
    }
    float m[3] = {-INFINITY, -INFINITY, -INFINITY}, z[3] = {}, o[3][4] = {};
    for (int s0 = j0; s0 < j1; s0 += SUB) {
        int s1 = min(j1, s0 + SUB);
        // スコア: ワープがキーを 1 つずつ受け持つ。q.k / sqrt(HD) と相対位置バイアスを、ワープの和で一度に求める
        #pragma unroll 4
        for (int j = s0 + warp; j < s1; j += 4) {
            float4 k = h4(kc + kv_row(j, ring) + kvh * HD + lane * 4);
            int d = pos - j;
            float rp = lane < NREL && d < EXTENT ? rel_proj[lane * EXTENT + d] : 0.f;
            #pragma unroll
            for (int h = 0; h < 3; h++) {
                float p = warp_sum((q[h].x * k.x + q[h].y * k.y + q[h].z * k.z + q[h].w * k.w) * rsqrtf((float)HD) + rl[h] * rp);
                if (lane == 0) sc[h][j - s0] = p;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int h = 0; h < 3; h++) {
            float bm = -INFINITY;
            for (int i = t; i < s1 - s0; i += blockDim.x) bm = fmaxf(bm, sc[h][i]);
            float mn = fmaxf(m[h], block_max(bm)), f = m[h] == -INFINITY ? 0.f : expf(m[h] - mn), zs = 0.f;
            for (int i = t; i < s1 - s0; i += blockDim.x) zs += sc[h][i] = expf(sc[h][i] - mn);
            z[h] = z[h] * f + block_sum(zs);
            #pragma unroll
            for (int c = 0; c < 4; c++) o[h][c] *= f;
            m[h] = mn;
        }
        // 値: ワープがキーを分け、レーンが 4 次元ずつ積和する
        #pragma unroll 8
        for (int i = warp; i < s1 - s0; i += 4) {
            float4 v = h4(vc + kv_row(s0 + i, ring) + kvh * HD + lane * 4);
            #pragma unroll
            for (int h = 0; h < 3; h++) {
                float p = sc[h][i];
                o[h][0] += p * v.x, o[h][1] += p * v.y, o[h][2] += p * v.z, o[h][3] += p * v.w;
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int h = 0; h < 3; h++)
        #pragma unroll
        for (int c = 0; c < 4; c++) osum[warp][h][lane * 4 + c] = o[h][c];
    __syncthreads();
    #pragma unroll
    for (int h = 0; h < 3; h++) {
        float* out = part + ((tt * 24 + kvh * 3 + h) * NSPLIT + split) * PART;
        out[2 + t] = osum[0][h][t] + osum[1][h][t] + osum[2][h][t] + osum[3][h][t];
        if (t == 0) out[0] = m[h], out[1] = z[h];
    }
    __shared__ bool last;
    __threadfence();
    __syncthreads();
    if (t == 0) last = atomicAdd(done + tt * 24 + kvh, 1u) == NSPLIT - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (t == 0) done[tt * 24 + kvh] = 0;
    #pragma unroll
    for (int h = 0; h < 3; h++) attn_combine(qx, part, ob, kvh * 3 + h, tt);
}

// (prefill) 注意の共通部分。グリッドは (KV ヘッド 8, T/AQ)、128スレッド。スレッドはトークン tid/AP の、KV を共有する 3 ヘッドの次元 (tid%AP)*AD から AD 個を受け持つ。
// q と o をレジスタに置き、共有メモリから読んだ K, V の AD 次元を 3 ヘッドに使い回す (共有メモリの帯域が律速になるため)
#define AP 4
#define AD (HD / AP)
#define AQ (128 / AP)
#define AK 16

// q と (最大値 m, 指数和 z, 重み付き和 o) を用意する。first なら sink から始め、そうでなければ acc [T][24][HD + 2] の続きから
__device__ __forceinline__ void attn_init(const float* qx, const float* acc, int first, int t0, int nq, int kvh, float qr[3][AD], float o[3][AD], float* m, float* z) {
    int qi = threadIdx.x / AP, part = threadIdx.x % AP;
    bool valid = qi < nq;
    const float* qh = qx + (size_t)(t0 + qi) * QX;
    #pragma unroll
    for (int h = 0; h < 3; h++) {
        int head = kvh * 3 + h;
        const float* a = first ? nullptr : acc + ((size_t)(t0 + qi) * 24 + head) * PART;
        m[h] = !valid ? 0.f : first ? qh[SINK_OFF + head] : a[0];
        z[h] = !valid || first ? 1.f : a[1];
        #pragma unroll
        for (int d = 0; d < AD; d++) {
            qr[h][d] = valid ? qh[head * HD + part * AD + d] : 0.f;
            o[h][d] = !valid || first ? 0.f : a[2 + part * AD + d];
        }
    }
}

// 位置 jlo..=jmax のキーの分を (m, z, o) に足す。行 j の KV は kb, vb の kv_row(j - b0, ring) にある
__device__ __forceinline__ void attn_keys(const float* qx, const half_t* kb, const half_t* vb, int b0, int ring, int jlo, int jmax, const float* rel_proj, int t0, int nq,
                                          int p0, int kvh, const float qr[3][AD], float o[3][AD], float* m, float* z) {
    __shared__ __align__(16) float Ks[AK][HD];
    __shared__ __align__(16) float Vs[AK][HD];
    __shared__ float Ps[AQ][3][AK];
    __shared__ float Rs[AQ][3][NREL];
    int tid = threadIdx.x, qi = tid / AP, part = tid % AP, pq = p0 + qi;
    bool valid = qi < nq;
    for (int i = tid; i < AQ * 3 * NREL; i += blockDim.x) {
        int q = i / (3 * NREL), h = i / NREL % 3;
        Rs[q][h][i % NREL] = q < nq ? qx[(size_t)(t0 + q) * QX + R_OFF + (kvh * 3 + h) * NREL + i % NREL] : 0.f;
    }
    for (int j0 = jlo; j0 <= jmax; j0 += AK) {
        __syncthreads();
        for (int i = tid; i < AK * HD; i += blockDim.x) {
            int k = i / HD, j = j0 + k;
            bool in = j <= jmax;
            Ks[k][i % HD] = in ? h2f(kb[kv_row(j - b0, ring) + kvh * HD + i % HD]) : 0.f;
            Vs[k][i % HD] = in ? h2f(vb[kv_row(j - b0, ring) + kvh * HD + i % HD]) : 0.f;
        }
        __syncthreads();
        float tmax[3] = {-INFINITY, -INFINITY, -INFINITY};
        for (int k = 0; k < AK; k++) {
            float ac[3] = {};
            #pragma unroll
            for (int e = 0; e < AD; e += 4) {
                float4 y = *(const float4*)&Ks[k][part * AD + e];
                #pragma unroll
                for (int h = 0; h < 3; h++) ac[h] += qr[h][e] * y.x + qr[h][e + 1] * y.y + qr[h][e + 2] * y.z + qr[h][e + 3] * y.w;
            }
            int j = j0 + k, d = pq - j;
            bool ok = valid && j <= jmax && j <= pq && (!ring || d < EXTENT);
            #pragma unroll
            for (int h = 0; h < 3; h++) {
                float x = ac[h] + __shfl_xor_sync(0xffffffffu, ac[h], 1);
                x += __shfl_xor_sync(0xffffffffu, x, 2);
                float s = ok ? x * rsqrtf((float)HD) + rel_bias(Rs[qi][h], rel_proj, d) : -INFINITY;
                tmax[h] = fmaxf(tmax[h], s);
                if (k % AP == part) Ps[qi][h][k] = s;
            }
        }
        #pragma unroll
        for (int h = 0; h < 3; h++) {
            float mn = fmaxf(m[h], tmax[h]), f = expf(m[h] - mn), zs = 0.f;
            for (int k = part; k < AK; k += AP) {
                float p = expf(Ps[qi][h][k] - mn);
                Ps[qi][h][k] = p;
                zs += p;
            }
            zs += __shfl_xor_sync(0xffffffffu, zs, 1);
            zs += __shfl_xor_sync(0xffffffffu, zs, 2);
            z[h] = z[h] * f + zs;
            m[h] = mn;
            #pragma unroll
            for (int d = 0; d < AD; d++) o[h][d] *= f;
        }
        __syncwarp();
        for (int k = 0; k < AK; k++) {
            float p0_ = Ps[qi][0][k], p1 = Ps[qi][1][k], p2 = Ps[qi][2][k];
            #pragma unroll
            for (int e = 0; e < AD; e += 4) {
                float4 v = *(const float4*)&Vs[k][part * AD + e];
                o[0][e] += p0_ * v.x, o[0][e + 1] += p0_ * v.y, o[0][e + 2] += p0_ * v.z, o[0][e + 3] += p0_ * v.w;
                o[1][e] += p1 * v.x, o[1][e + 1] += p1 * v.y, o[1][e + 2] += p1 * v.z, o[1][e + 3] += p1 * v.w;
                o[2][e] += p2 * v.x, o[2][e + 1] += p2 * v.y, o[2][e + 2] += p2 * v.z, o[2][e + 3] += p2 * v.w;
            }
        }
    }
}

// (prefill、長い文脈の global 層) VRAM に送った KV の塊 (位置 b0 から nb 個、kb [nb][1024]、vb 同じ) の分だけ注意を計算し、
// acc [T][24][HD + 2] の (最大値, 指数和, 重み付き和) に足す。first なら sink から始める
extern "C" __global__ void attn_blk(const float* qx, const half_t* kb, const half_t* vb, int b0, int nb, const float* rel_proj, float* acc,
                                    const int* posp, int T, int first) {
    PDL();
    int kvh = blockIdx.x, qi = threadIdx.x / AP, part = threadIdx.x % AP;
    int t0 = blockIdx.y * AQ, nq = min(AQ, T - t0), p0 = *posp + t0;
    __align__(16) float qr[3][AD], o[3][AD];
    float m[3], z[3];
    attn_init(qx, acc, first, t0, nq, kvh, qr, o, m, z);
    attn_keys(qx, kb, vb, b0, 0, b0, min(b0 + nb - 1, p0 + nq - 1), rel_proj, t0, nq, p0, kvh, qr, o, m, z);
    if (qi < nq) {
        #pragma unroll
        for (int h = 0; h < 3; h++) {
            float* a = acc + ((size_t)(t0 + qi) * 24 + kvh * 3 + h) * PART;
            if (part == 0) a[0] = m[h], a[1] = z[h];
            #pragma unroll
            for (int d = 0; d < AD; d++) a[2 + part * AD + d] = o[h][d];
        }
    }
}

// attn_blk の結果を正規化し、出力ゲートをかける。グリッドは (ヘッド, T)、128スレッド
extern "C" __global__ void attn_fin(const float* qx, const float* acc, float* out) {
    PDL();
    int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const float* a = acc + ((size_t)t * 24 + h) * PART;
    out[(size_t)t * 3072 + h * HD + d] = a[2 + d] / a[1] * sigmoidf(qx[(size_t)t * QX + GATE_OFF + h * HD + d]);
}

// (T <= 8 用) 部分結果と sink をまとめ、出力ゲートをかける。グリッドは (ヘッド, T)、128スレッド
// attn_part の最後のブロックが呼ぶ。NSPLIT 個の部分結果を sink と合わせて、ゲートをかけて out に書く。part は L1 を通さずに読む
__device__ void attn_combine(const float* qx, const float* part, float* out, int h, int tt) {
    int t = threadIdx.x;
    qx += (size_t)tt * QX;
    out += (size_t)tt * 3072;
    const float* p = part + (tt * 24 + h) * NSPLIT * PART;
    float sink = qx[SINK_OFF + h], m = sink;
    for (int s = 0; s < NSPLIT; s++) m = fmaxf(m, __ldcg(p + s * PART));
    float z = expf(sink - m), o = 0.f;
    for (int s = 0; s < NSPLIT; s++) {
        float ps = __ldcg(p + s * PART), pz = __ldcg(p + s * PART + 1);
        float f = pz > 0.f ? expf(ps - m) : 0.f;
        z += pz * f;
        o += __ldcg(p + s * PART + 2 + t) * f;
    }
    out[h * HD + t] = o / z * sigmoidf(qx[GATE_OFF + h * HD + t]);
}

// (prefill、VRAM の KV) 相対位置のバイアス rel_proj[:, d] . r を、トークンとヘッドごとに d < EXTENT の全部について relb [T][24][EXTENT] に書く。
// グリッドは (T, 24)
extern "C" __global__ void rel_table(const float* qx, const float* rel_proj, float* relb, int T) {
    PDL();
    __shared__ float r[NREL];
    int t = blockIdx.x, h = blockIdx.y;
    if (threadIdx.x < NREL) r[threadIdx.x] = qx[(size_t)t * QX + R_OFF + h * NREL + threadIdx.x];
    __syncthreads();
    for (int d = threadIdx.x; d < EXTENT; d += blockDim.x) {
        float s = 0.f;
        #pragma unroll
        for (int i = 0; i < NREL; i++) s += r[i] * rel_proj[i * EXTENT + d];
        relb[((size_t)t * 24 + h) * EXTENT + d] = s;
    }
}

// (prefill、VRAM の KV) 注意をテンソルコア (f16 入力、f32 累積) で計算し、出力ゲートをかけて out [T][3072] に書く。ring > 0 なら SWA 層で、直近 EXTENT 個の位置を見る。
// グリッドは (KV ヘッド 8, T/32)、192スレッド。ワープ w はヘッド kvh*3 + w%3 の、トークン 16 個 (t0 + (w/3)*16 から) を受け持つ。
// キーを 64 個ずつ共有メモリに載せ、KV を共有する 3 ヘッドで使い回す。確率は書き出さずに、行ごとの最大値と指数和を更新しながら V と掛ける
#define FK 32
#define FST (HD + 8) // 共有メモリの行の長さ (f16)。フラグメントを読むときにバンクが衝突しない
extern "C" __global__ void attn_fa(const float* qx, const half_t* kc, const half_t* vc, const float* relb, float* out, const int* posp, int ring, int T) {
    PDL();
    __shared__ __align__(16) half_t Ks[FK * FST], Vs[FK * FST];
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    int kvh = blockIdx.x, head = kvh * 3 + warp % 3, t0 = blockIdx.y * 32, nq = min(32, T - t0);
    int pos = *posp, p0 = pos + t0, jlo = ring ? max(0, p0 + 1 - EXTENT) : 0, jmax = p0 + nq - 1;
    int tq[2] = {t0 + warp / 3 * 16 + g, t0 + warp / 3 * 16 + g + 8};
    bool qok[2] = {tq[0] < T, tq[1] < T};
    // q (rsqrt(HD) を掛けておく) の A フラグメント。行 g, g+8、次元 16ks + 2t (+8)
    unsigned qa[8][4];
    const float sc = rsqrtf((float)HD);
    #pragma unroll
    for (int ks = 0; ks < 8; ks++) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int r = i & 1, dim = ks * 16 + (i >> 1) * 8 + 2 * t;
            float2 v = qok[r] ? *(const float2*)(qx + (size_t)tq[r] * QX + head * HD + dim) : make_float2(0.f, 0.f);
            qa[ks][i] = h2pack(v.x * sc, v.y * sc);
        }
    }
    float sink = qx[(size_t)min(tq[0], T - 1) * QX + SINK_OFF + head], sink1 = qx[(size_t)min(tq[1], T - 1) * QX + SINK_OFF + head];
    float m[2] = {sink, sink1}, z[2] = {t == 0 ? 1.f : 0.f, t == 0 ? 1.f : 0.f}, o[16][4] = {};
    const float* rb[2] = {relb + ((size_t)min(tq[0], T - 1) * 24 + head) * EXTENT, relb + ((size_t)min(tq[1], T - 1) * 24 + head) * EXTENT};
    // キーの塊は、前の塊の積和の前にレジスタへ読んでおく。スレッドは塊の 16 バイト単位 tid + 192q (q < FQ) を受け持つ
    constexpr int FQ = (FK * HD / 8 + 191) / 192;
    uint4 kr[FQ], vr[FQ];
    auto fetch = [&](int j0) {
        #pragma unroll
        for (int q = 0; q < FQ; q++) {
            int i = tid + 192 * q, k = i / (HD / 8), c = i % (HD / 8) * 8, j = j0 + k;
            kr[q] = vr[q] = make_uint4(0, 0, 0, 0);
            if (i < FK * HD / 8 && j <= jmax) {
                kr[q] = *(const uint4*)(kc + kv_row(j, ring) + kvh * HD + c);
                vr[q] = *(const uint4*)(vc + kv_row(j, ring) + kvh * HD + c);
            }
        }
    };
    fetch(jlo);
    for (int j0 = jlo; j0 <= jmax; j0 += FK) {
        __syncthreads();
        #pragma unroll
        for (int q = 0; q < FQ; q++) {
            int i = tid + 192 * q, k = i / (HD / 8), c = i % (HD / 8) * 8;
            if (i < FK * HD / 8) {
                *(uint4*)&Ks[k * FST + c] = kr[q];
                *(uint4*)&Vs[k * FST + c] = vr[q];
            }
        }
        __syncthreads();
        if (j0 + FK <= jmax) fetch(j0 + FK);
        float s[FK / 8][4] = {};
        #pragma unroll
        for (int ks = 0; ks < 8; ks++) {
            #pragma unroll
            for (int nt = 0; nt < FK / 8; nt += 2) {
                unsigned b[4];
                ldb((const unsigned*)Ks, FST, nt * 8, ks * 16, b);
                mma_f16(s[nt], qa[ks], b[0], b[1]);
                mma_f16(s[nt + 1], qa[ks], b[2], b[3]);
            }
        }
        float mx[2] = {-INFINITY, -INFINITY};
        #pragma unroll
        for (int nt = 0; nt < FK / 8; nt++) {
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                int r = c >> 1, j = j0 + nt * 8 + 2 * t + (c & 1), d = pos + tq[r] - j;
                bool ok = qok[r] && j <= jmax && d >= 0 && (!ring || d < EXTENT);
                float v = ok ? s[nt][c] + (d < EXTENT ? rb[r][d] : 0.f) : -INFINITY;
                s[nt][c] = v;
                mx[r] = fmaxf(mx[r], v);
            }
        }
        float f[2];
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
            float mn = fmaxf(m[r], mx[r]);
            f[r] = __expf(m[r] - mn);
            m[r] = mn;
            z[r] *= f[r];
        }
        #pragma unroll
        for (int nt = 0; nt < FK / 8; nt++) {
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                float p = __expf(s[nt][c] - m[c >> 1]);
                s[nt][c] = p;
                z[c >> 1] += p;
            }
        }
        #pragma unroll
        for (int dn = 0; dn < 16; dn++) o[dn][0] *= f[0], o[dn][1] *= f[0], o[dn][2] *= f[1], o[dn][3] *= f[1];
        #pragma unroll
        for (int kk = 0; kk < FK / 16; kk++) {
            unsigned pa[4] = {h2pack(s[2 * kk][0], s[2 * kk][1]), h2pack(s[2 * kk][2], s[2 * kk][3]), h2pack(s[2 * kk + 1][0], s[2 * kk + 1][1]), h2pack(s[2 * kk + 1][2], s[2 * kk + 1][3])};
            #pragma unroll
            for (int dn = 0; dn < 16; dn += 2) {
                // V [キー][次元] を転置して読む: 行列 0,1 は次元 8dn のキー 0..7, 8..15、行列 2,3 は次元 8(dn+1)
                const half_t* a = &Vs[(kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * FST + (dn + (lane >> 4)) * 8];
                unsigned b0, b1, b2, b3, addr = (unsigned)__cvta_generic_to_shared(a);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(addr));
                mma_f16(o[dn], pa, b0, b1);
                mma_f16(o[dn + 1], pa, b2, b3);
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        z[r] += __shfl_xor_sync(0xffffffffu, z[r], 1);
        z[r] += __shfl_xor_sync(0xffffffffu, z[r], 2);
    }
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        if (!qok[r]) continue;
        const float* gate = qx + (size_t)tq[r] * QX + GATE_OFF + head * HD;
        float* dst = out + (size_t)tq[r] * 3072 + head * HD;
        #pragma unroll
        for (int dn = 0; dn < 16; dn++) {
            int dim = dn * 8 + 2 * t;
            dst[dim] = o[dn][2 * r] / z[r] * sigmoidf(gate[dim]);
            dst[dim + 1] = o[dn][2 * r + 1] / z[r] * sigmoidf(gate[dim + 1]);
        }
    }
}

extern "C" __global__ void add(const float* a, const float* b, float* out, int n) {
    PDL();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

// logits に softcap をかけ、最大の位置 (同じ値なら番号の小さい方) を best に残す (先に 0 にしておく)。
// 値の大小がそのまま 64 ビットの大小になるよう、上位 32 ビットを並べ替えた値、下位を 0xffffffff - 位置にして atomicMax をとる
extern "C" __global__ void softcap(float* x, int n, unsigned long long* best, float* host) {
    PDL();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long key = 0;
    if (i < n) {
        float v = 30.f * tanhf(x[i] / 30.f);
        x[i] = v;
        if (host) {
            host[i] = v; // 主記憶にも写す (生成)。CPU は done_host の旗を見てから読む
            __threadfence_system();
        }
        unsigned u = __float_as_uint(v);
        u = u & 0x80000000u ? ~u : u | 0x80000000u;
        key = (unsigned long long)u << 32 | (0xffffffffu - (unsigned)i);
    }
    for (int o = 16; o > 0; o >>= 1) key = max(key, __shfl_xor_sync(0xffffffffu, key, o));
    if ((threadIdx.x & 31) == 0 && key) atomicMax(best, key);
}

// ルーター: sigmoid、バイアス込みで上位8個を選び、重みを正規化して route_scale (8) を掛ける。ブロックはトークンごとで、先頭の1ワープが計算する。
// レーンは expert lane + 32j (j = 0..3) を持ち、ワープ内で最大を8回求める。同じ値なら番号の小さい方を選ぶ (CPU 版と同じ)
// 1ワープで呼ぶ。logits は他のブロックが書いたものでもよい (L1 を通さずに読む)
__device__ void topk_warp(const float* logits, const float* bias, int* ids, float* wts, int lane) {
    float p[4], s[4], w[TOPK], sum = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        p[j] = sigmoidf(__ldcg(logits + lane + 32 * j));
        s[j] = p[j] + bias[lane + 32 * j];
    }
    #pragma unroll
    for (int k = 0; k < TOPK; k++) {
        float bv = s[0];
        int bi = lane;
        #pragma unroll
        for (int j = 1; j < 4; j++)
            if (s[j] > bv) bv = s[j], bi = lane + 32 * j;
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
        }
        float pj = p[0];
        #pragma unroll
        for (int j = 1; j < 4; j++)
            if (bi >> 5 == j) pj = p[j];
        w[k] = __shfl_sync(0xffffffffu, pj, bi & 31);
        sum += w[k];
        if (lane == (bi & 31)) {
            #pragma unroll
            for (int j = 0; j < 4; j++)
                if (bi >> 5 == j) s[j] = -INFINITY;
        }
        if (lane == k) ids[k] = bi;
    }
    if (lane < TOPK) {
        float wk = w[0];
        #pragma unroll
        for (int k = 1; k < TOPK; k++)
            if (lane == k) wk = w[k];
        wts[lane] = wk / (sum + 1e-20f) * 8.f;
    }
}

extern "C" __global__ void topk(const float* logits, const float* bias, int* ids, float* wts) {
    PDL();
    if (threadIdx.x < 32) topk_warp(logits + blockIdx.x * NEXP, bias, ids + blockIdx.x * TOPK, wts + blockIdx.x * TOPK, threadIdx.x);
}

// NVFP4 の 16 要素ブロック1つ (pk) と x[0..16] (xv) の内積。scale は掛けない
__device__ __forceinline__ float fp4_block(uint2 pk, const float4* xv) {
    float b = 0.f;
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        unsigned u = half ? pk.y : pk.x;
        float4 x0 = xv[half * 2], x1 = xv[half * 2 + 1];
        b += e2m1(u) * x0.x + e2m1(u >> 4) * x0.y + e2m1(u >> 8) * x0.z + e2m1(u >> 12) * x0.w;
        b += e2m1(u >> 16) * x1.x + e2m1(u >> 20) * x1.y + e2m1(u >> 24) * x1.z + e2m1(u >> 28) * x1.w;
    }
    return b;
}

// 生成の MoE は、レーンが 16 要素ブロック lane + 32j を受け持ち、ブロックごとに b * scale を足してからワープで和をとる。
// 1ワープが複数の行を受け持ち、読んだ x のブロックを全部の行で使う (行ごとに x を読み直すと L1 の処理が律速になる)。各行の足す順は1行ずつのときと同じ

// (T <= 8 用) 選ばれた8 expert の gate と up を計算して SITU をかける。1ワープが、割り当て a = t*8+k の i0..i0+3 番目
// (gate と up が交互に並んだ連続 8 行) を受け持つ
extern "C" __global__ void moe_up_nvfp4(const u8* w13, const u8* s13, const float* a13, const int* ids, const float* x, float* act, int T) {
    int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (gw >= T * TOPK * FF / 4) return;
    PDL();
    int a = gw / (FF / 4), i0 = gw % (FF / 4) * 4, ex = ids[a];
    const float* xt = x + (size_t)(a / TOPK) * D;
    size_t row0 = (size_t)ex * 2 * FF + 2 * i0;
    const u8* w = w13 + row0 * (D / 2);
    const u8* s = s13 + row0 * (D / 16);
    float acc[8] = {};
    #pragma unroll 1
    for (int j = 0; j < (D / 16 + 31) / 32; j++) {
        int blk = lane + 32 * j;
        if (blk >= D / 16) break;
        const float4* xx = (const float4*)(xt + blk * 16);
        float4 xv[4] = {xx[0], xx[1], xx[2], xx[3]};
        uint2 pk[8];
        unsigned sc[8];
        #pragma unroll
        for (int r = 0; r < 8; r++) pk[r] = __ldg((const uint2*)(w + r * (D / 2) + blk * 8)), sc[r] = __ldg(s + r * (D / 16) + blk);
        #pragma unroll
        for (int r = 0; r < 8; r++) acc[r] += fp4_block(pk[r], xv) * e4m3(sc[r]);
    }
    float sca = a13[ex] / 2688.f;
    #pragma unroll
    for (int r = 0; r < 8; r += 2) {
        float g = warp_sum(acc[r]), u = warp_sum(acc[r + 1]);
        if (lane == 0) act[(size_t)a * FF + i0 + r / 2] = situ(g * sca, u * sca);
    }
}

// (T <= 8 用) down 射影と8 expert の重み付き和。1ワープが、トークン t の出力の連続 2 行 r0, r0+1 を受け持つ
extern "C" __global__ void moe_down_nvfp4(const u8* w2, const u8* s2, const float* a2, const int* ids, const float* wts, const float* act, float* y, int T) {
    int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (gw >= T * D / 2) return;
    int t = gw / (D / 2), r0 = gw % (D / 2) * 2;
    PDL();
    float acc[2] = {};
    #pragma unroll 2
    for (int k = 0; k < TOPK; k++) {
        int a = t * TOPK + k, ex = ids[a];
        const float* xk = act + (size_t)a * FF;
        size_t rr = (size_t)ex * D + r0;
        const u8* w = w2 + rr * (FF / 2);
        const u8* s = s2 + rr * (FF / 16);
        float d[2] = {};
        #pragma unroll
        for (int j = 0; j < (FF / 16 + 31) / 32; j++) {
            int blk = lane + 32 * j;
            if (blk >= FF / 16) break;
            const float4* xx = (const float4*)(xk + blk * 16);
            float4 xv[4] = {xx[0], xx[1], xx[2], xx[3]};
            #pragma unroll
            for (int r = 0; r < 2; r++)
                d[r] += fp4_block(__ldg((const uint2*)(w + r * (FF / 2) + blk * 8)), xv) * e4m3(__ldg(s + r * (FF / 16) + blk));
        }
        #pragma unroll
        for (int r = 0; r < 2; r++) acc[r] += d[r] * wts[a] * a2[ex] / 2688.f;
    }
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        float v = warp_sum(acc[r]);
        if (lane == 0) y[(size_t)t * D + r0 + r] = v;
    }
}

// (T <= 8 用) y[t][r] = sum_c W[r][c] x[t][c]。W は BF16。重みを1回読むあいだに T 本と積和する。1行を1ワープで計算する
extern "C" __global__ void mv_bf16_t(const bf16* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, int rows, int cols, int T) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= rows) return;
    const uint4* wr = (const uint4*)(w + (size_t)row * cols);
    // 重みは前のカーネルに依存しないので、待つあいだに L2 へ読み込んでおく
    for (int i = lane * 8; i < cols / 8; i += 256) asm volatile("prefetch.global.L2 [%0];" :: "l"(wr + i));
    PDL();
    float acc[8] = {};
    #pragma unroll 2
    for (int i = lane; i < cols / 8; i += 32) {
        uint4 v = __ldg(wr + i);
        float wf[8] = {bf(v.x & 0xffffu), bf(v.x >> 16), bf(v.y & 0xffffu), bf(v.y >> 16), bf(v.z & 0xffffu), bf(v.z >> 16), bf(v.w & 0xffffu), bf(v.w >> 16)};
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            if (t >= T) break;
            const float4* xx = (const float4*)(x + (size_t)t * cols + i * 8);
            float4 a = xx[0], b = xx[1];
            acc[t] += wf[0] * a.x + wf[1] * a.y + wf[2] * a.z + wf[3] * a.w + wf[4] * b.x + wf[5] * b.y + wf[6] * b.z + wf[7] * b.w;
        }
    }
    #pragma unroll
    for (int t = 0; t < 8; t++) {
        if (t >= T) break;
        float s = warp_sum(acc[t]);
        if (lane == 0) y[(size_t)t * rows + row] = s;
    }
}

// (prefill 用) T*8 個の割り当て (トークン t の k 番目 = t*8+k) を expert 順に並べる。order[off[e]..off[e+1]] が expert e の割り当て。
// 1ブロック 1024スレッド。expert の中の並びは決まらないが、割り当ての計算はどれも独立なので結果は変わらない
extern "C" __global__ void moe_sort(const int* ids, int n, int* order, int* off) {
    PDL();
    __shared__ int cnt[NEXP], at[NEXP];
    int tid = threadIdx.x;
    for (int e = tid; e < NEXP; e += blockDim.x) cnt[e] = 0;
    __syncthreads();
    for (int a = tid; a < n; a += blockDim.x) atomicAdd(&cnt[ids[a]], 1);
    __syncthreads();
    if (tid == 0) {
        int s = 0;
        for (int e = 0; e < NEXP; e++) off[e] = at[e] = s, s += cnt[e];
        off[NEXP] = s;
    }
    __syncthreads();
    for (int a = tid; a < n; a += blockDim.x) order[atomicAdd(&at[ids[a]], 1)] = a;
}

// (prefill 用) expert ごとに、その expert に割り当てられた入力行をまとめて掛ける。重みはバッチにつき1回だけ読む
// Y[a][r] = amax/2688 * sum_c W_e[r][c] X[a / xdiv][c]。グリッドは (rows/64, expert)、128スレッド
// e2m1 x e4m3 は BF16 で誤差なく表せるので、amax/2688 は最後に掛ける
extern "C" __global__ void gemm_nvfp4_grouped(const u8* __restrict__ w, const u8* __restrict__ s, const float* amax, int rows, int cols,
                                              const float* __restrict__ x, int xdiv, const int* order, const int* off, float* y) {
    PDL();
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
    PDL();
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

// SQ 形式 (Strata の S2_K, S3_K, SF4) の expert。行列1つのレコード (gate/up は1行、down は2行の組) は 2304 要素で、16 要素のブロック 144 個に分ける。
// S_K の平面: lo [レコード][144] (u32。2ビット x 16)、hi [レコード][144] (u16。S3 の3ビット目 x 16)、sc は u [レコード][36] (u8) のあとに D [レコード][9] (f16)。
// 重みは (q - half) * D * u / 255 で、u は 64 要素ごと、D は 256 要素ごと。SF4 の平面: lo [レコード][144] (u64)、sc [レコード][144] (E4M3)
struct SkMat {
    unsigned long long lo, sc;
    unsigned code;
    float g; // SF4 の大域スケール。S_K は 1
};

// 平面の並び (strata.rs の split_sk): lo, hi は [4 ブロックの塊][レコード][4]、u は [64 要素の組][レコード]、D は [256 要素の組][レコード]
__device__ __forceinline__ size_t sk_blk(int nrec, int rec, int blk) { return ((size_t)(blk / 4) * nrec + rec) * 4 + blk % 4; }

__device__ __forceinline__ float sk_scale(const u8* sc, int nrec, int rec, int g) {
    const half_t* d = (const half_t*)(sc + (size_t)nrec * 36) + (size_t)(g / 4) * nrec + rec;
    return h2f(__ldg(d)) * ((float)__ldg(sc + (size_t)g * nrec + rec) / 255.f);
}

// S_K の16要素のブロック (lo: 2ビット x 16、hi: S3 の3ビット目 x 16) を、要素 (n, n+8) の組の f16 x2 8個にする。値は (q - half) / 8。
// f16 の 64 は仮数部の1目盛りが 1/16 なので、仮数部に 2q を置くと 64 + q/8 になる。そこから 64 + half/8 を引く
__device__ __forceinline__ void sk_unpack(unsigned lo, unsigned hi, bool s3, unsigned v[8]) {
    unsigned h2 = (hi & 0xffu) | ((hi & 0xff00u) << 8), bias = s3 ? 0x54075407u : 0x54035403u;
    #pragma unroll
    for (int n = 0; n < 8; n++) {
        unsigned x = (n == 0 ? lo << 1 : lo >> (2 * n - 1)) & 0x00060006u;
        if (s3) x |= (n <= 3 ? h2 << (3 - n) : h2 >> (n - 3)) & 0x00080008u;
        asm("sub.f16x2 %0, %1, %2;" : "=r"(v[n]) : "r"(x | 0x54005400u), "r"(bias));
    }
}

// S_K の16要素のブロックを、要素 (n, n+8) の組の f16 x2 8個にする。q を仮数部にそのまま置くので、値は q * 2^-24 (非正規化数) で丸めはない。
// half を引く分は、使う側で活性の和から引く
__device__ __forceinline__ void sk_unpack_q(unsigned lo, unsigned hi, bool s3, unsigned v[8]) {
    unsigned h2 = ((hi & 0xffu) | ((hi & 0xff00u) << 8)) << 2;
    #pragma unroll
    for (int n = 0; n < 8; n++) v[n] = ((lo >> (2 * n)) & 0x00030003u) | (s3 ? (h2 >> n) & 0x00040004u : 0u);
}

// (T <= 8 用) SQ 形式の expert の行列とベクトルの積をテンソルコアで計算する。同じ expert を選んだトークンは、重みを1回読んでまとめて掛ける。
// グリッドは (T*8, rows/128)、128スレッド。ブロックは割り当て a の expert の 128行を受け持つ (それより前の割り当てと同じ expert なら何もしない)。
// ワープは 32行を、K の全部について mma (16行 x K16 x 8列 を2つ) で計算する。列 n はトークン n で、その expert を選んでいなければ 0 になる。
// x は f16_perm の並びの f16 で、which = 0 (gate/up) なら行はトークン、1 (down) なら割り当て。y[割り当て][rows] に m.g を掛けて書く。
// レーン l は行 l の 4 ブロック (64 要素、u 1つ分) を読んで f16 に展開し、ワープの共有メモリ [32行][64] に置く。テンソルコアの入力は ldmatrix で読む。
// 次の 4 ブロック分は、今の分の積和の前に読んでおく
#define SKB_ST 72
extern "C" __global__ void moe_skb(const SkMat* tab, int which, int rows, int cols, const unsigned* __restrict__ x, const int* ids, int T, float* y) {
    PDL();
    __shared__ __align__(16) half_t Wsm[4][32 * SKB_ST];
    __shared__ float Ssm[4][32];
    int a = blockIdx.x, e = ids[a];
    for (int i = 0; i < a; i++)
        if (ids[i] == e) return;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const SkMat m = tab[e * 2 + which];
    bool fp4 = m.code == 15, s3 = m.code == 3;
    int nrec = which ? D / 2 : 2 * FF;
    // トークン n がこの expert を選んだ割り当て。選んでいなければ -1
    auto asg = [&](int n) {
        int r = -1;
        if (n < T)
            for (int k = 0; k < TOPK; k++)
                if (ids[n * TOPK + k] == e) r = n * TOPK + k;
        return r;
    };
    int ag = asg(g), ac[2] = {asg(2 * t), asg(2 * t + 1)};
    const unsigned* xr = ag < 0 ? nullptr : x + (size_t)(which ? ag : g) * (cols / 2);
    int r0 = blockIdx.y * 128 + warp * 32, row = r0 + lane, rec = row >> which, b0r = (row & which) * (cols / 16);
    // k4 (4 の倍数) からの 4 ブロックの位置は sk_blk(nrec, rec, b0r + k4) から連続する
    // 4 ブロック分の重み、スケールの元の値 (u と D)、活性 (B フラグメント)。2 つ先の分まで読んでおき (c1, c2)、スケールは使うときに計算する
    struct Chunk {
        uint4 L[2];
        uint2 H;
        unsigned SC, X[8];
        unsigned short U, SD;
    };
    auto fetch = [&](int k4) {
        Chunk c{};
        if (k4 >= cols / 16) return c;
        size_t at = sk_blk(nrec, rec, b0r + k4);
        if (fp4) {
            c.L[0] = __ldg((const uint4*)((const uint2*)m.lo + at));
            c.L[1] = __ldg((const uint4*)((const uint2*)m.lo + at + 2));
            c.SC = __ldg((const unsigned*)((const u8*)m.sc + at));
        } else {
            c.L[0] = __ldg((const uint4*)((const unsigned*)m.lo + at));
            if (s3) c.H = __ldg((const uint2*)((const unsigned short*)(m.lo + (size_t)nrec * 576) + at));
            int gi = (b0r + k4) / 4;
            c.U = __ldg((const u8*)m.sc + (size_t)gi * nrec + rec);
            c.SD = __ldg((const unsigned short*)((const u8*)m.sc + (size_t)nrec * 36) + (size_t)(gi / 4) * nrec + rec);
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) c.X[2 * i] = xr ? __ldg(xr + (k4 + i) * 8 + t) : 0u, c.X[2 * i + 1] = xr ? __ldg(xr + (k4 + i) * 8 + t + 4) : 0u;
        return c;
    };
    Chunk c1 = fetch(0), c2 = fetch(4);
    float acc[2][4] = {};
    half_t* W = Wsm[warp];
    // seg は q * 2^-24 との積なので、スケールに 2^24 を掛けておき、half は 2^-24 倍で引く
    float hq = fp4 ? 0.f : (s3 ? 3.5f : 1.5f) / 16777216.f;
    for (int k4 = 0; k4 < cols / 16; k4 += 4) {
        Chunk c0 = c1;
        c1 = c2;
        c2 = fetch(k4 + 8);
        const uint4* L = c0.L;
        uint2 H = c0.H;
        unsigned SC = c0.SC;
        __syncwarp();
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            unsigned v[8];
            if (fp4) {
                // E2M1 の3ビット (指数2, 仮数1) を f16 の指数の下2ビットと仮数の先頭に置くと 2^-14 倍の値になる。ブロックのスケール (E4M3) の 16 倍を f16 で掛ける
                unsigned px = c & 1 ? (c & 2 ? L[1].z : L[0].z) : (c & 2 ? L[1].x : L[0].x);
                unsigned py = c & 1 ? (c & 2 ? L[1].w : L[0].w) : (c & 2 ? L[1].y : L[0].y);
                unsigned sc = h2pack(e4m3((SC >> (8 * c)) & 0xffu) * 16.f, e4m3((SC >> (8 * c)) & 0xffu) * 16.f);
                #pragma unroll
                for (int n = 0; n < 8; n++) {
                    unsigned q = ((px >> (4 * n)) & 0xfu) | (((py >> (4 * n)) & 0xfu) << 16);
                    unsigned f = ((q & 0x00070007u) << 9) | ((q & 0x00080008u) << 12);
                    asm("mul.f16x2 %0, %1, %2;" : "=r"(v[n]) : "r"(f), "r"(sc));
                }
            } else {
                unsigned l = c == 0 ? L[0].x : c == 1 ? L[0].y : c == 2 ? L[0].z : L[0].w;
                sk_unpack_q(l, s3 ? ((c & 2 ? H.y : H.x) >> (16 * (c & 1))) & 0xffffu : 0u, s3, v);
            }
            uint4* dst = (uint4*)(W + lane * SKB_ST + c * 16);
            dst[0] = make_uint4(v[0], v[1], v[2], v[3]);
            dst[1] = make_uint4(v[4], v[5], v[6], v[7]);
        }
        Ssm[warp][lane] = fp4 ? 1024.f : 16777216.f * h2f(c0.SD) * ((float)c0.U / 255.f); // SF4 の値は 2^-10 倍で展開する
        __syncwarp();
        // xs は列ごとの活性の和 (全部 1 の行列との積)。S_K では q * 2^-24 との積から half * (活性の和) を引く
        float seg[2][4] = {}, xs[4] = {};
        const unsigned ones[4] = {0x3c003c00u, 0x3c003c00u, 0x3c003c00u, 0x3c003c00u};
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            unsigned b0 = c0.X[2 * i], b1 = c0.X[2 * i + 1];
            if (!fp4) mma_f16(xs, ones, b0, b1);
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                unsigned A[4];
                ldsm4(W + (h * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * SKB_ST + i * 16 + (lane >> 4) * 8, A);
                mma_f16(seg[h], A, b0, b1);
            }
        }
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            float sa = Ssm[warp][h * 16 + g], sb = Ssm[warp][h * 16 + g + 8];
            #pragma unroll
            for (int c = 0; c < 4; c++) acc[h][c] += (seg[h][c] - hq * xs[c]) * (c < 2 ? sa : sb);
        }
    }
    #pragma unroll
    for (int h = 0; h < 2; h++)
        #pragma unroll
        for (int c = 0; c < 2; c++)
            if (ac[c] >= 0) {
                y[(size_t)ac[c] * rows + r0 + h * 16 + g] = acc[h][c] * m.g;
                y[(size_t)ac[c] * rows + r0 + h * 16 + g + 8] = acc[h][2 + c] * m.g;
            }
}

// (prefill 用) SQ 形式の expert の行列積。y[a] = W_e x[a / xdiv] (x は f16_perm で並べ替えた f16) を、expert e に割り当てた a (order[off[e]..off[e+1]]) について求める。
// グリッドは (rows/64, 128 * SKP)、128スレッド。ブロックは 64行 x 割り当て 32 個ずつを受け持ち (expert の割り当ての 32 個の組を SKP 個のブロックで分ける)、K は 64 ずつ (S_K の u の1単位) 共有メモリに載せる。
// テンソルコアは f16 入力、f32 累積。16要素のブロックの中の並びを、重みと活性の両方で (0, 8, 1, 9, ...) に入れ替えてある (積和は変わらない)。
// 次の K の分は、今の分の積和の前にレジスタへ読んでおく
#define SKP 8
// 16要素のブロックの中の並び (0, 8, 1, 9, ...) で、要素 j の位置
__device__ __forceinline__ int perm16(int j) { int n = j & 15; return (j & ~15) + (n < 8 ? 2 * n : 2 * n - 15); }

// which = 0 (gate/up) では、出力にそのまま SITU と ixd をかけ、down の入力 act [割り当て][FF] (f16_perm の並びの f16) に書く (situ_sk と同じ値)
extern "C" __global__ void gemm_sk_grouped(const SkMat* tab, int which, int rows, int cols, const uint4* __restrict__ x, int xdiv, const int* order, const int* off, float* y, half_t* act, const float* ixd) {
    PDL();
    __shared__ __align__(16) unsigned Ws[64 * WST / 2], Xs[32 * WST / 2];
    __shared__ int as[32];
    int e = blockIdx.y / SKP, beg = off[e] + blockIdx.y % SKP * 32, end = off[e + 1]; // 同じ expert のブロックを並べ、重みを L2 で共有する
    if (beg >= end) return;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    int r0 = blockIdx.x * 64, nrec = which ? D / 2 : 2 * FF;
    const SkMat m = tab[e * 2 + which];
    bool fp4 = m.code == 15, s3 = m.code == 3;
    // 重みはスレッドあたり 16要素のブロック2つ (行 tid/2、ブロック (tid%2)*2 + i)、活性は割り当て tid/4 の 16要素 (ブロック tid%4)
    int wr = tid / 2, xj = tid / 4, xb = tid % 4;
    int rrow = r0 + wr, rec = rrow >> which, blk0 = (rrow & which) * (cols / 16);
    for (int c0 = beg; c0 < end; c0 += 32 * SKP) {
        __syncthreads();
        if (tid < 32) as[tid] = c0 + tid < end ? order[c0 + tid] : -1;
        __syncthreads();
        int nj = (min(end - c0, 32) + 7) / 8; // 割り当てのある 8列の組の数
        int xr = as[xj] >= 0 ? as[xj] / xdiv : -1;
        float acc[4][4] = {};
        uint2 wl[2];
        unsigned wh[2];
        uint4 xv[2];
        float nsc[2] = {1.f, 1.f};
        auto fetch = [&](int k0) {
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                int blk = blk0 + k0 / 16 + (tid % 2) * 2 + i;
                if (fp4) {
                    wl[i] = ((const uint2*)m.lo)[sk_blk(nrec, rec, blk)];
                    wh[i] = ((const u8*)m.sc)[sk_blk(nrec, rec, blk)];
                } else {
                    wl[i].x = ((const unsigned*)m.lo)[sk_blk(nrec, rec, blk)];
                    wh[i] = s3 ? ((const unsigned short*)(m.lo + (size_t)nrec * 576))[sk_blk(nrec, rec, blk)] : 0u;
                }
            }
            #pragma unroll
            for (int q = 0; q < 2; q++) xv[q] = xr >= 0 ? x[((size_t)xr * cols + k0 + xb * 16) / 8 + q] : make_uint4(0, 0, 0, 0);
            if (!fp4) {
                #pragma unroll
                for (int h = 0; h < 2; h++) {
                    int row = r0 + warp * 16 + g + h * 8;
                    nsc[h] = 8.f * sk_scale((const u8*)m.sc, nrec, row >> which, (row & which) * (cols / 64) + k0 / 64);
                }
            }
        };
        fetch(0);
        for (int k0 = 0; k0 < cols; k0 += 64) {
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                unsigned v[8];
                if (fp4) {
                    float sc = e4m3(wh[i]);
                    #pragma unroll
                    for (int n = 0; n < 8; n++) v[n] = h2pack(e2m1(wl[i].x >> (4 * n)) * sc, e2m1(wl[i].y >> (4 * n)) * sc);
                } else {
                    sk_unpack(wl[i].x, wh[i], s3, v);
                }
                unsigned* dst = Ws + wr * (WST / 2) + ((tid % 2) * 2 + i) * 8;
                *(uint4*)dst = make_uint4(v[0], v[1], v[2], v[3]);
                *(uint4*)(dst + 4) = make_uint4(v[4], v[5], v[6], v[7]);
            }
            *(uint4*)(Xs + xj * (WST / 2) + xb * 8) = xv[0];
            *(uint4*)(Xs + xj * (WST / 2) + xb * 8 + 4) = xv[1];
            __syncthreads();
            float sc[2] = {nsc[0], nsc[1]};
            int kn = k0 + 64;
            if (kn < cols) fetch(kn);
            float seg[4][4] = {};
            #pragma unroll
            for (int ks = 0; ks < 4; ks++) {
                int r = warp * 16 + g, k = ks * 8 + t;
                unsigned a[4] = {Ws[r * (WST / 2) + k], Ws[(r + 8) * (WST / 2) + k], Ws[r * (WST / 2) + k + 4], Ws[(r + 8) * (WST / 2) + k + 4]};
                #pragma unroll
                for (int j = 0; j < 4; j++)
                    if (j < nj) mma_f16(seg[j], a, Xs[(j * 8 + g) * (WST / 2) + k], Xs[(j * 8 + g) * (WST / 2) + k + 4]);
            }
            // u は 64 要素ごとなので、K の1単位ごとに行のスケールを掛けて足す (S_K の値は 1/8 にしてある)
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                acc[j][0] += seg[j][0] * sc[0], acc[j][1] += seg[j][1] * sc[0];
                acc[j][2] += seg[j][2] * sc[1], acc[j][3] += seg[j][3] * sc[1];
            }
        }
        if (which == 0) {
            // 行 2i (gate) と 2i+1 (up) は、g が隣り合うレーン (lane ^ 4) にある
            #pragma unroll
            for (int j = 0; j < 4; j++)
                #pragma unroll
                for (int h = 0; h < 2; h++) {
                    float a[2] = {acc[j][2 * h] * m.g, acc[j][2 * h + 1] * m.g}, b[2];
                    #pragma unroll
                    for (int q = 0; q < 2; q++) b[q] = __shfl_xor_sync(0xffffffffu, a[q], 4);
                    if (g & 1) continue;
                    int jj = (r0 + warp * 16 + g + 8 * h) / 2;
                    #pragma unroll
                    for (int q = 0; q < 2; q++) {
                        int c = j * 8 + 2 * t + q;
                        if (as[c] >= 0) act[(size_t)as[c] * FF + perm16(jj)] = f2h(situ(a[q], b[q]) * ixd[(size_t)e * FF + jj]);
                    }
                }
        } else {
            store_y(acc, r0, [&](int r, int c, float v) {
                if (as[c] >= 0) y[(size_t)as[c] * rows + r] = v * m.g;
            });
        }
    }
}

// (prefill 用) x [n] を f16 にして、16要素のブロックの中を perm16 の並びにする (gemm_sk_grouped の入力)
extern "C" __global__ void f16_perm(const float* x, half_t* out, int n) {
    PDL();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[perm16(i)] = f2h(x[i]);
}

// (prefill 用) situ_rows の SQ 版。活性を expert の列ごとの逆スケールで割る
// 出力は f16_perm と同じ並びの f16
extern "C" __global__ void situ_sk(const float* gu, half_t* act, const int* ids, const float* ixd, int n) {
    PDL();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * FF) return;
    int a = i / FF, j = i % FF;
    act[perm16(i)] = f2h(situ(gu[(size_t)a * 2 * FF + 2 * j], gu[(size_t)a * 2 * FF + 2 * j + 1]) * ixd[(size_t)ids[a] * FF + j]);
}

// (prefill 用) order[base..base+n] の割り当てだけ SITU をかける
extern "C" __global__ void situ_sel(const float* gu, float* act, const int* order, int base, int n) {
    PDL();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * FF) return;
    int a = order[base + i / FF], j = i % FF;
    act[(size_t)a * FF + j] = situ(gu[(size_t)a * 2 * FF + 2 * j], gu[(size_t)a * 2 * FF + 2 * j + 1]);
}

// (prefill 用) gate/up が交互に並ぶ [n][2FF] に SITU をかけて [n][FF] にする
extern "C" __global__ void situ_rows(const float* gu, float* act, int n) {
    PDL();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * FF) return;
    int a = i / FF, j = i % FF;
    act[i] = situ(gu[(size_t)a * 2 * FF + 2 * j], gu[(size_t)a * 2 * FF + 2 * j + 1]);
}

// (prefill 用) y[t] = sum_k wts[t][k] yd[t*8+k]
extern "C" __global__ void moe_sum(const float* yd, const float* wts, float* y, int T) {
    PDL();
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
    PDL();
    const float* seq = (const float*)tab[3 * blockIdx.y];
    float* hist = (float*)tab[3 * blockIdx.y + 1];
    int ch = (int)tab[3 * blockIdx.y + 2], c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    for (int k = 0; k < 3; k++) hist[c * 3 + k] = seq[(size_t)(a + k) * ch + c];
}

// 行ごとの、先頭 n 個のうち最大の値の位置 (同じ値なら小さい位置)。行は stride 個おき。ブロックは行ごと、1024スレッド
extern "C" __global__ void argmax_rows(const float* x, int stride, int n, int* out) {
    PDL();
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

// ---- 下書きモデル (DSpark、mtp.safetensors) ----
// 重みは読み込み時に行ごとのスケール付き FP8 (E4M3) にする。下書きは本体が必ず確かめるので、精度は受け入れ率にしか効かない

#define DQKV 5120   // q 3072 + k 1024 + v 1024
#define DRAFT_CTX 8192 // 下書きモデルが扱う文脈の長さ (global 層の KV を VRAM に置く)
#define DFF 9216

__device__ __forceinline__ unsigned char f2e4m3(float x) {
    unsigned short r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(0.f), "f"(x));
    return (unsigned char)(r & 0xff);
}

// BF16 の行を、行の最大の絶対値が 448 になるスケールで FP8 にする。ブロックは行ごと、256スレッド
extern "C" __global__ void quant_rows_bf16(const bf16* src, int cols, u8* dst, float* scale) {
    PDL();
    const bf16* r = src + (size_t)blockIdx.x * cols;
    float m = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) m = fmaxf(m, fabsf(bf(r[c])));
    m = block_max(m);
    float s = m > 0.f ? m / 448.f : 1.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) dst[(size_t)blockIdx.x * cols + c] = f2e4m3(bf(r[c]) / s);
    if (threadIdx.x == 0) scale[blockIdx.x] = s;
}

// y [k][ld] の列 v (長さ k) を FP8 の行 v にする。ブロックは列ごと、k (= 256) スレッド
extern "C" __global__ void quant_cols_f32(const float* y, int ld, int k, u8* dst, float* scale) {
    PDL();
    int v = blockIdx.x, j = threadIdx.x;
    float x = j < k ? y[(size_t)j * ld + v] : 0.f;
    float m = block_max(fabsf(x));
    float s = m > 0.f ? m / 448.f : 1.f;
    if (j < k) dst[(size_t)v * k + j] = f2e4m3(x / s);
    if (j == 0) scale[v] = s;
}

// y[t][r] = scale[r] * sum_c W[r][c] x[t][c] (t < T <= 8)。W は FP8 の [rows, cols]、cols は 16 の倍数、rows は 4 の倍数。
// 1ワープが 4 行を受け持ち、活性の読み込みを 4 行で使い回す
extern "C" __global__ void mv_fp8(const u8* __restrict__ w, const float* __restrict__ sc, const float* __restrict__ x, float* __restrict__ y,
                                  int rows, int cols, int T) {
    PDL();
    int row0 = ((blockIdx.x * blockDim.x + threadIdx.x) >> 5) * 4, lane = threadIdx.x & 31;
    if (row0 >= rows) return;
    float acc[4][8] = {};
    for (int i = lane; i < cols / 16; i += 32) {
        float wf[4][16];
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            uint4 v = __ldg((const uint4*)(w + (size_t)(row0 + r) * cols) + i);
            unsigned u[4] = {v.x, v.y, v.z, v.w};
            #pragma unroll
            for (int k = 0; k < 16; k++) wf[r][k] = e4m3(u[k / 4] >> (8 * (k % 4)));
        }
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            if (t >= T) break;
            const float4* xx = (const float4*)(x + (size_t)t * cols + i * 16);
            #pragma unroll
            for (int q = 0; q < 4; q++) {
                float4 a = xx[q];
                #pragma unroll
                for (int r = 0; r < 4; r++) acc[r][t] += wf[r][4 * q] * a.x + wf[r][4 * q + 1] * a.y + wf[r][4 * q + 2] * a.z + wf[r][4 * q + 3] * a.w;
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < 4; r++)
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            if (t >= T) break;
            float s = warp_sum(acc[r][t]);
            if (lane == 0) y[(size_t)t * rows + row0 + r] = s * sc[row0 + r];
        }
}
// out[t] = sum_j a_j taps[j][t] (taps の種類は tstride 要素おき)
extern "C" __global__ void tapmix(const float* taps, int tstride, float a0, float a1, float a2, float a3, int n, float* out) {
    PDL();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * D) return;
    out[i] = a0 * taps[i] + a1 * taps[tstride + i] + a2 * taps[2 * tstride + i] + a3 * taps[3 * tstride + i];
}

// 128 次元のヘッド x をヘッドごとの RMS 正規化 (重み w) してから、位置 pos の NeoX 式の RoPE (base 1e6) をかける。128スレッド
__device__ float head_norm_rope(float x, const float* w, int pos) {
    __shared__ float sh[HD];
    int d = threadIdx.x;
    x *= rsqrtf(block_sum(x * x) / HD + 1e-6f) * w[d];
    sh[d] = x;
    __syncthreads();
    int i = d % 64;
    float s, c;
    sincosf((float)pos * powf(1e6f, -2.f * i / HD), &s, &c);
    float r = d < 64 ? sh[d] * c - sh[d + 64] * s : sh[d - 64] * s + sh[d] * c;
    __syncthreads();
    return r;
}

// 文脈の K/V: kv [n][2048] (k 1024, v 1024) の K を正規化と RoPE、位置 pos0+t の行に f16 で書く。ring > 0 ならリングバッファ。
// グリッドは (KV ヘッド 8, n)、128スレッド
extern "C" __global__ void dr_ctx_store(const float* kv, const float* knorm, int pos0, int ring, half_t* kc, half_t* vc) {
    PDL();
    int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x, pos = pos0 + t;
    float k = head_norm_rope(kv[(size_t)t * 2048 + h * HD + d], knorm, pos);
    size_t at = (size_t)(ring ? pos % ring : pos) * 1024 + h * HD + d;
    kc[at] = f2h(k);
    vc[at] = f2h(kv[(size_t)t * 2048 + 1024 + h * HD + d]);
}

// ブロック 8 行の q と k (qkv [8][5120]) を正規化と RoPE (位置 pos0+行)。グリッドは (q 24 + k 8, 8)、128スレッド
extern "C" __global__ void dr_qk(float* qkv, const float* qn, const float* kn, int pos0) {
    PDL();
    int hh = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    float* p = qkv + (size_t)t * DQKV + (hh < 24 ? hh * HD : 3072 + (hh - 24) * HD);
    float r = head_norm_rope(p[d], hh < 24 ? qn : kn, pos0 + t);
    p[d] = r;
}

// ブロック 8 行の注意。キーは文脈の位置 lo..hi (キャッシュ) とブロックの 8 行で、マスクなし。
// グリッドは (KV ヘッド 8, 分割数)、128スレッド。ブロックは KV ヘッド 1つの 24 行 (ヘッド 3つ x 8 行) を、キーの DR_SPLIT 個ずつの区間について計算し、
// 部分結果 (最大値, 指数和, 重み付き和) を part [分割][KV ヘッド][24][HD + 2] に書く。dr_fin がまとめる。
// ワープ w は行 6w..6w+5 を受け持ち、スコアはレーン = キー、重み付き和はレーン = 次元 4 個
#define DR_SPLIT 256
#define DR_K 32
extern "C" __global__ void dr_attn(const float* qkv, const half_t* kc, const half_t* vc, int ring, int lo, int hi, float* part) {
    PDL();
    __shared__ float Qs[24][HD], Ks[DR_K][HD + 1], Vs[DR_K][HD], Ps[24][DR_K];
    int kvh = blockIdx.x, s = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int nc = hi - lo, n = nc + 8, j0 = s * DR_SPLIT, j1 = min(n, j0 + DR_SPLIT);
    for (int i = tid; i < 24 * HD; i += blockDim.x) {
        int r = i / HD, d = i % HD; // 行 r = ヘッド r/8 の、ブロックの行 r%8
        Qs[r][d] = qkv[(size_t)(r % 8) * DQKV + (kvh * 3 + r / 8) * HD + d] * rsqrtf((float)HD);
    }
    float m[6], z[6], o[6][4];
    #pragma unroll
    for (int i = 0; i < 6; i++) {
        m[i] = -INFINITY, z[i] = 0.f;
        #pragma unroll
        for (int c = 0; c < 4; c++) o[i][c] = 0.f;
    }
    for (int k0 = j0; k0 < j1; k0 += DR_K) {
        __syncthreads();
        for (int i = tid; i < DR_K * HD; i += blockDim.x) {
            int k = i / HD, d = i % HD, j = k0 + k;
            float kv = 0.f, vv = 0.f;
            if (j < j1) {
                if (j < nc) {
                    size_t at = kv_row(lo + j, ring) + kvh * HD + d;
                    kv = h2f(kc[at]), vv = h2f(vc[at]);
                } else {
                    const float* b = qkv + (size_t)(j - nc) * DQKV;
                    kv = b[3072 + kvh * HD + d], vv = b[4096 + kvh * HD + d];
                }
            }
            Ks[k][d] = kv, Vs[k][d] = vv;
        }
        __syncthreads();
        bool ok = k0 + lane < j1;
        #pragma unroll
        for (int i = 0; i < 6; i++) {
            int r = warp * 6 + i;
            float sc = 0.f;
            #pragma unroll 8
            for (int d = 0; d < HD; d++) sc += Qs[r][d] * Ks[lane][d];
            sc = ok ? sc : -INFINITY;
            float mx = sc;
            for (int x = 16; x > 0; x >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, x));
            float mn = fmaxf(m[i], mx), f = __expf(m[i] - mn), p = ok ? __expf(sc - mn) : 0.f;
            float ps = p;
            for (int x = 16; x > 0; x >>= 1) ps += __shfl_xor_sync(0xffffffffu, ps, x);
            z[i] = z[i] * f + ps, m[i] = mn;
            #pragma unroll
            for (int c = 0; c < 4; c++) o[i][c] *= f;
            Ps[r][lane] = p;
        }
        __syncwarp();
        #pragma unroll
        for (int i = 0; i < 6; i++) {
            int r = warp * 6 + i;
            for (int k = 0; k < DR_K; k++) {
                float p = Ps[r][k];
                float4 v = *(const float4*)&Vs[k][lane * 4];
                o[i][0] += p * v.x, o[i][1] += p * v.y, o[i][2] += p * v.z, o[i][3] += p * v.w;
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < 6; i++) {
        float* out = part + (((size_t)s * 8 + kvh) * 24 + warp * 6 + i) * (HD + 2);
        if (lane == 0) out[0] = m[i], out[1] = z[i];
        #pragma unroll
        for (int c = 0; c < 4; c++) out[2 + lane * 4 + c] = o[i][c];
    }
}

// dr_attn の部分結果をまとめて out [8][3072] に書く。グリッドは (ヘッド 24, 行 8)、128スレッド
extern "C" __global__ void dr_fin(const float* part, int ns, float* out) {
    PDL();
    int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x, r = (h % 3) * 8 + t;
    float m = -INFINITY;
    for (int s = 0; s < ns; s++) m = fmaxf(m, part[(((size_t)s * 8 + h / 3) * 24 + r) * (HD + 2)]);
    float z = 0.f, o = 0.f;
    for (int s = 0; s < ns; s++) {
        const float* p = part + (((size_t)s * 8 + h / 3) * 24 + r) * (HD + 2);
        float f = p[1] > 0.f ? __expf(p[0] - m) : 0.f;
        z += p[1] * f, o += p[2 + d] * f;
    }
    out[(size_t)t * 3072 + h * HD + d] = o / z;
}

// act[t][j] = silu(gu[t][j]) * gu[t][DFF + j]
extern "C" __global__ void silu_mul(const float* gu, float* act, int T) {
    PDL();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * DFF) return;
    int t = i / DFF, j = i % DFF;
    float g = gu[(size_t)t * 2 * DFF + j];
    act[i] = g / (1.f + expf(-g)) * gu[(size_t)t * 2 * DFF + DFF + j];
}

// x2[t] = [hidden[t], frnn] (8 行)
extern "C" __global__ void concat_frnn(const float* hidden, const float* frnn, float* x2) {
    PDL();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 8 * 2 * D) return;
    int t = i / (2 * D), j = i % (2 * D);
    x2[i] = j < D ? hidden[(size_t)t * D + j] : frnn[j - D];
}

extern "C" __global__ void add_bias_rows(float* y, const float* b, int cols, int n) {
    PDL();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += b[i % cols];
}

// DSpark の Markov ヘッドの1手の前半: raw = ctxm + J01 [state; w1[prev]] (prev は prevp[0])。グリッドは 96、256 スレッド。ワープが raw の 1 行を受け持つ
extern "C" __global__ void markov_raw(const float* ctxm, const float* j01, const bf16* w1, const float* state, const int* prevp, float* raw) {
    PDL();
    __shared__ float in[512];
    int t = threadIdx.x, lane = t & 31, o = blockIdx.x * 8 + (t >> 5);
    in[t] = state[t];
    in[256 + t] = bf(w1[(size_t)*prevp * 256 + t]);
    __syncthreads();
    const float* j = j01 + (size_t)o * 512;
    float s = 0.f;
    #pragma unroll 4
    for (int k = lane; k < 512; k += 32) s += j[k] * in[k];
    s = warp_sum(s);
    if (lane == 0) raw[o] = s + ctxm[o];
}

// 後半: ゲートで state を更新し、latent = tanh(raw[512..768]) を lat に書く。1ブロック 256 スレッド
extern "C" __global__ void markov_step(const float* raw, float* state, float* lat) {
    PDL();
    int t = threadIdx.x;
    float gate = 1.f / (1.f + expf(-raw[t]));
    state[t] = gate * state[t] + (1.f - gate) * tanhf(raw[256 + t]);
    lat[t] = tanhf(raw[512 + t]);
}

// Markov ヘッドの1手: lg = fold x (fold は FP8 の [n][256]、行のスケール sc)、logits = softcap(lg + fb + base) の、ブロックごとの
// (最大値, その位置, sum exp(logits - 最大値)) を part [ブロック][3] に書く (fold_fin がまとめる)。
// 1行を 4 レーン (64 列ずつ) で計算し、ワープは連続する 8 行を読む。グリッドは n/64、256 スレッド
extern "C" __global__ void fold_mv(const u8* __restrict__ w, const float* __restrict__ sc, const float* __restrict__ x, const float* fb, const float* base, int n, float* part) {
    PDL();
    __shared__ float bv[8], bz[8];
    __shared__ int bi[8];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, q = lane & 3, r = (blockIdx.x * 8 + warp) * 8 + (lane >> 2);
    float s = 0.f;
    if (r < n) {
        const uint4* wr = (const uint4*)(w + (size_t)r * 256 + q * 64);
        const float4* xx = (const float4*)(x + q * 64);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint4 v = __ldg(wr + i);
            unsigned u[4] = {v.x, v.y, v.z, v.w};
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float4 a = xx[i * 4 + j];
                s += e4m3(u[j]) * a.x + e4m3(u[j] >> 8) * a.y + e4m3(u[j] >> 16) * a.z + e4m3(u[j] >> 24) * a.w;
            }
        }
    }
    s += __shfl_xor_sync(0xffffffffu, s, 1);
    s += __shfl_xor_sync(0xffffffffu, s, 2);
    float v = r < n && q == 0 ? 30.f * tanhf((s * sc[r] + fb[r] + base[r]) / 30.f) : -INFINITY;
    float m = v;
    int mi = r;
    for (int o = 16; o > 0; o >>= 1) {
        float v2 = __shfl_xor_sync(0xffffffffu, m, o);
        int i2 = __shfl_xor_sync(0xffffffffu, mi, o);
        if (v2 > m || (v2 == m && i2 < mi)) m = v2, mi = i2;
    }
    if (lane == 0) bv[warp] = m, bi[warp] = mi;
    __syncthreads();
    m = bv[0], mi = bi[0];
    for (int k = 1; k < 8; k++)
        if (bv[k] > m || (bv[k] == m && bi[k] < mi)) m = bv[k], mi = bi[k];
    float z = v == -INFINITY ? 0.f : expf(v - m);
    z = warp_sum(z);
    if (lane == 0) bz[warp] = z;
    __syncthreads();
    if (threadIdx.x == 0) {
        float zs = 0.f;
        for (int k = 0; k < 8; k++) zs += bz[k];
        part[blockIdx.x * 3] = m, part[blockIdx.x * 3 + 1] = __int_as_float(mi), part[blockIdx.x * 3 + 2] = zs;
    }
}

// fold_part の nb 個の結果をまとめ、最大の位置 (同じ値なら小さい位置) と sum exp(logits - 最大値) を書く。1ブロック 256 スレッド
extern "C" __global__ void fold_fin(const float* part, int nb, int* best, float* sumexp) {
    PDL();
    float v = -INFINITY;
    int i = 0x7fffffff;
    for (int b = threadIdx.x; b < nb; b += blockDim.x) {
        float v2 = part[b * 3];
        int i2 = __float_as_int(part[b * 3 + 1]);
        if (v2 > v || (v2 == v && i2 < i)) v = v2, i = i2;
    }
    for (int o = 16; o > 0; o >>= 1) {
        float v2 = __shfl_xor_sync(0xffffffffu, v, o);
        int i2 = __shfl_xor_sync(0xffffffffu, i, o);
        if (v2 > v || (v2 == v && i2 < i)) v = v2, i = i2;
    }
    __shared__ float bv[8];
    __shared__ int bi[8];
    if ((threadIdx.x & 31) == 0) bv[threadIdx.x >> 5] = v, bi[threadIdx.x >> 5] = i;
    __syncthreads();
    v = bv[0], i = bi[0];
    for (int w = 1; w < 8; w++)
        if (bv[w] > v || (bv[w] == v && bi[w] < i)) v = bv[w], i = bi[w];
    float z = 0.f;
    for (int b = threadIdx.x; b < nb; b += blockDim.x) z += part[b * 3 + 2] * expf(part[b * 3] - v);
    z = block_sum(z);
    if (threadIdx.x == 0) *best = i, *sumexp = z;
}

// ---- 生成時の 20〜23層 (MXFP8) の expert の VRAM キャッシュ ----
// よく選ばれる expert を VRAM の枠に置いておき、選ばれたら GPU で計算する (残りは CPU が同時に計算する)。
// 枠は1つの expert 分 [w13 | s13 | w2 | s2] で、枠 j は cache + j * XSLOT。slots は使う枠の番号を 8 ビットずつ並べたもの (最大 8 個)
#define XW13 (2 * FF * D)
#define XS13 (XW13 / 32)
#define XW2 (D * FF)
#define XSLOT (XW13 + XS13 + XW2 + XW2 / 32)

// MXFP8 の1行 (n 要素、32 要素ごとに UE8M0 の scale) と x の内積の部分和。レーンは 16 要素ずつ分担する
__device__ __forceinline__ float mx_row(const u8* w, const u8* s, const float* x, int n, int lane) {
    float acc = 0.f;
    for (int c = lane; c < n / 16; c += 32) {
        uint4 pk = __ldg((const uint4*)(w + c * 16));
        const float4* xx = (const float4*)(x + c * 16);
        unsigned u[4] = {pk.x, pk.y, pk.z, pk.w};
        float b = 0.f;
        #pragma unroll
        for (int q = 0; q < 4; q++) {
            float4 v = xx[q];
            b += e4m3(u[q]) * v.x + e4m3(u[q] >> 8) * v.y + e4m3(u[q] >> 16) * v.z + e4m3(u[q] >> 24) * v.w;
        }
        acc += b * __uint_as_float((unsigned)s[c / 2] << 23);
    }
    return acc;
}

// 枠の expert の gate と up に SITU をかけ、act [n][FF] に書く。使う枠の数 n、枠の番号 (8 ビットずつ)、重みは route にある
// (route: n u32、+8 に slots u64、+16 に重み f32 x 8。CPU が主記憶に書き、wait_host が VRAM に写したもの)。1ワープが j 番目の expert の行 2i (gate) と 2i+1 (up) を受け持つ
extern "C" __global__ void moe_up_mx(const u8* cache, const u8* route, const float* x, float* act) {
    PDL();
    int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    int n = *(const int*)route;
    if (gw >= n * FF) return;
    unsigned long long slots = *(const unsigned long long*)(route + 8);
    int j = gw / FF, i = gw % FF;
    const u8* w13 = cache + (size_t)((slots >> (8 * j)) & 255) * XSLOT;
    const u8* s13 = w13 + XW13;
    size_t row = 2 * (size_t)i;
    float g = warp_sum(mx_row(w13 + row * D, s13 + row * (D / 32), x, D, lane));
    float u = warp_sum(mx_row(w13 + (row + 1) * D, s13 + (row + 1) * (D / 32), x, D, lane));
    if (lane == 0) act[gw] = situ(g, u);
}

// 枠の expert の down 射影の、重みつきの和。1ワープが出力の1行を受け持つ
extern "C" __global__ void moe_down_mx(const u8* cache, const u8* route, const float* act, float* y) {
    PDL();
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= D) return;
    int n = *(const int*)route;
    unsigned long long slots = *(const unsigned long long*)(route + 8);
    float acc = 0.f;
    for (int j = 0; j < n; j++) {
        const u8* w2 = cache + (size_t)((slots >> (8 * j)) & 255) * XSLOT + XW13 + XS13;
        const u8* s2 = w2 + XW2;
        acc += mx_row(w2 + (size_t)row * FF, s2 + (size_t)row * (FF / 32), act + j * FF, FF, lane) * ((const float*)(route + 16))[j];
    }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}

// (2 <= T <= 8、検証) MXFP8 の1行と cnt 本の入力 xp[t] の内積の部分和を acc[t] に足す。レーンは 16 要素ずつ分担し、重みは1回だけ読む
__device__ __forceinline__ void mx_row_t(const u8* w, const u8* s, const float* const* xp, int cnt, int n, int lane, float* acc) {
    for (int c = lane; c < n / 16; c += 32) {
        uint4 pk = __ldg((const uint4*)(w + c * 16));
        float sc = __uint_as_float((unsigned)s[c / 2] << 23);
        unsigned u[4] = {pk.x, pk.y, pk.z, pk.w};
        float wf[16];
        #pragma unroll
        for (int q = 0; q < 4; q++) wf[q * 4] = e4m3(u[q]), wf[q * 4 + 1] = e4m3(u[q] >> 8), wf[q * 4 + 2] = e4m3(u[q] >> 16), wf[q * 4 + 3] = e4m3(u[q] >> 24);
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            if (t >= cnt) break;
            const float4* xx = (const float4*)(xp[t] + c * 16);
            float b = 0.f;
            #pragma unroll
            for (int q = 0; q < 4; q++) {
                float4 v = xx[q];
                b += wf[q * 4] * v.x + wf[q * 4 + 1] * v.y + wf[q * 4 + 2] * v.z + wf[q * 4 + 3] * v.w;
            }
            acc[t] += b * sc;
        }
    }
}

// (検証) 枠にある expert の gate と up に SITU をかける。off[u]..off[u+1] が u 番目の expert の割り当て (order)、off[65 + u] がその枠。
// 1ワープが1つの expert の行 2i (gate) と 2i+1 (up) を、その expert を選んだ全トークンについて計算する
extern "C" __global__ void moe_up_mx_g(const u8* cache, const int* order, const int* off, const float* x, float* act) {
    PDL();
    int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    int u = gw / FF, i = gw % FF;
    if (u >= 64) return;
    int slot = off[65 + u];
    if (slot < 0) return;
    int beg = off[u], cnt = off[u + 1] - beg, as[8];
    const float* xp[8];
    #pragma unroll
    for (int c = 0; c < 8; c++) {
        as[c] = c < cnt ? order[beg + c] : 0;
        xp[c] = x + (size_t)(as[c] / TOPK) * D;
    }
    const u8* w13 = cache + (size_t)slot * XSLOT;
    const u8* s13 = w13 + XW13;
    size_t row = 2 * (size_t)i;
    float g[8] = {}, up[8] = {};
    mx_row_t(w13 + row * D, s13 + row * (D / 32), xp, cnt, D, lane, g);
    mx_row_t(w13 + (row + 1) * D, s13 + (row + 1) * (D / 32), xp, cnt, D, lane, up);
    #pragma unroll
    for (int c = 0; c < 8; c++) {
        if (c >= cnt) break;
        float gv = warp_sum(g[c]), uv = warp_sum(up[c]);
        if (lane == 0) act[(size_t)as[c] * FF + i] = situ(gv, uv);
    }
}

// (検証) 枠にある expert の down 射影を yd [T * 8][D] に書く (重みは moe_sum が掛ける)。1ワープが1つの expert の1行を受け持つ
extern "C" __global__ void moe_down_mx_g(const u8* cache, const int* order, const int* off, const float* act, float* yd) {
    PDL();
    int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    int u = gw / D, row = gw % D;
    if (u >= 64) return;
    int slot = off[65 + u];
    if (slot < 0) return;
    int beg = off[u], cnt = off[u + 1] - beg, as[8];
    const float* xp[8];
    #pragma unroll
    for (int c = 0; c < 8; c++) {
        as[c] = c < cnt ? order[beg + c] : 0;
        xp[c] = act + (size_t)as[c] * FF;
    }
    const u8* w2 = cache + (size_t)slot * XSLOT + XW13 + XS13;
    const u8* s2 = w2 + XW2;
    float acc[8] = {};
    mx_row_t(w2 + (size_t)row * FF, s2 + (size_t)row * (FF / 32), xp, cnt, FF, lane, acc);
    #pragma unroll
    for (int c = 0; c < 8; c++) {
        if (c >= cnt) break;
        float v = warp_sum(acc[c]);
        if (lane == 0) yd[(size_t)as[c] * D + row] = v;
    }
}

// ---- 生成時の CPU との受け渡し ----
// 20〜23層の MoE は、1トークン分の CUDA Graph の中で CPU と受け渡す。場所は GPU から直接読み書きできる主記憶で、
// 旗 (u32) で待ち合わせる。旗は書いた側が 1 にし、待っていた側が 0 に戻す。主記憶の値はキャッシュを通さずに読む

// x [D] を主記憶の out に写し、旗を立てる。1 ブロック
extern "C" __global__ void to_host(const float* x, float* out, unsigned* flag) {
    PDL();
    for (int i = threadIdx.x; i < D; i += blockDim.x) out[i] = x[i];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) *(volatile unsigned*)flag = 1;
}

// 旗が立つまで待ち、0 に戻す。n (<= 32) が 0 でなければ、続けて主記憶の src から u32 を n 個 dst (VRAM) に写す。1 ワープ
// (主記憶を多くのスレッドが直接読むと、PCIe の小さい読み出しが大量に出て遅い)
extern "C" __global__ void wait_host(unsigned* flag, const unsigned* src, unsigned* dst, int n) {
    PDL();
    if (threadIdx.x == 0) {
        while (*(volatile unsigned*)flag == 0) __nanosleep(200);
        *(volatile unsigned*)flag = 0;
        __threadfence_system();
    }
    __syncwarp();
    if (threadIdx.x < n) dst[threadIdx.x] = __ldcv(src + threadIdx.x);
}

// out = a + 主記憶の b
extern "C" __global__ void add_host(const float* a, const float* b, float* out) {
    PDL();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D) out[i] = a[i] + __ldcv(b + i);
}

// (生成、主記憶に KV がある global 層) 注意は CPU で計算する。[q 3072, r 384, sink 24] を主記憶の out に写し、旗を立てる。1 ブロック
extern "C" __global__ void attn_to_host(const float* qx, float* out, unsigned* flag) {
    PDL();
    for (int i = threadIdx.x; i < R_OFF + 384; i += blockDim.x) out[i] = qx[i];
    if (threadIdx.x < 24) out[R_OFF + 384 + threadIdx.x] = qx[SINK_OFF + threadIdx.x];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) *(volatile unsigned*)flag = 1;
}

// CPU が主記憶に書いた注意の結果 ho [3072] に出力ゲートをかけて out に書く。12 ブロック
extern "C" __global__ void attn_from_host(const float* qx, const float* ho, float* out) {
    PDL();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = __ldcv(ho + i) * sigmoidf(qx[GATE_OFF + i]);
}

// (2 <= T <= 8 用) ctrl_down と ctrl_up の、T トークンをまとめて計算する版。control j の入力は cur + j * cs (cs = 0 なら共通)、
// メッセージは msg + j * T * D、出力は out + j * T * D で、どれも [T][D]。g は [j][T][64]。重みは1回読んで T 本と積和する

// g[j][t][r] = down_j[r] . [rmsnorm(cur_j[t]), rmsnorm(msg_j[t])] (r < 64)。グリッドは (8, n)、256 スレッド
extern "C" __global__ void ctrl_down_t(const float* cur, int cs, const float* msg, const bf16* w0, const bf16* w1, const bf16* w2, float* g, int T) {
    PDL();
    __shared__ float kc[8], km[8], part[8][16];
    int jc = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    cur += (size_t)jc * cs;
    msg += (size_t)jc * T * D;
    const bf16* w = jc == 0 ? w0 : jc == 1 ? w1 : w2;
    // 正規化の係数: 全トークンの入力とメッセージの二乗和を1回の走査で求める
    float ss[16] = {};
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        #pragma unroll
        for (int t = 0; t < 8; t++)
            if (t < T) {
                float a = cur[(size_t)t * D + i], b = msg[(size_t)t * D + i];
                ss[2 * t] += a * a, ss[2 * t + 1] += b * b;
            }
    #pragma unroll
    for (int q = 0; q < 16; q++) {
        float v = warp_sum(ss[q]);
        if (lane == 0) part[warp][q] = v;
    }
    __syncthreads();
    if (threadIdx.x < 2 * T) {
        float v = 0.f;
        for (int w = 0; w < 8; w++) v += part[w][threadIdx.x];
        float k = rsqrtf(v / D + 1e-6f);
        if (threadIdx.x & 1) km[threadIdx.x / 2] = k;
        else kc[threadIdx.x / 2] = k;
    }
    __syncthreads();
    int row = blockIdx.x * 8 + warp;
    const uint4* wr = (const uint4*)(w + (size_t)row * 2 * D);
    float acc[8] = {};
    for (int i = lane; i < 2 * D / 8; i += 32) {
        uint4 v = __ldg(wr + i);
        float wf[8] = {bf(v.x & 0xffffu), bf(v.x >> 16), bf(v.y & 0xffffu), bf(v.y >> 16), bf(v.z & 0xffffu), bf(v.z >> 16), bf(v.w & 0xffffu), bf(v.w >> 16)};
        int h = i >= D / 8;
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            if (t >= T) break;
            const float4* xx = (const float4*)((h ? msg : cur) + (size_t)t * D + (i * 8 - h * D));
            float4 a = xx[0], b = xx[1];
            float k = h ? km[t] : kc[t];
            acc[t] += k * (wf[0] * a.x + wf[1] * a.y + wf[2] * a.z + wf[3] * a.w + wf[4] * b.x + wf[5] * b.y + wf[6] * b.z + wf[7] * b.w);
        }
    }
    #pragma unroll
    for (int t = 0; t < 8; t++) {
        if (t >= T) break;
        float s = warp_sum(acc[t]);
        if (lane == 0) g[((size_t)jc * T + t) * 64 + row] = s;
    }
}

// 1ワープが要素 i について、control j の gain と forget をレーン 8j..8j+7 で、トークンごとに求めて out_j[t][i] を書く。
// 重みの行は1回だけ読む。入力を読んでから書くので、out が cur と重なってよい
extern "C" __global__ void ctrl_up_t(const float* cur, int cs, float* out, const float* msg, const bf16* w0, const bf16* w1, const bf16* w2, const float* g, int n, int T) {
    PDL();
    int i = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31, jc = lane >> 3, l8 = lane & 7;
    if (i >= D) return;
    const bf16* w = jc == 0 ? w0 : jc == 1 ? w1 : w2;
    float wf[2][8] = {};
    if (jc < n)
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            uint4 u = __ldg((const uint4*)(w + (size_t)(h * D + i) * 64) + l8);
            float f[8] = {bf(u.x & 0xffffu), bf(u.x >> 16), bf(u.y & 0xffffu), bf(u.y >> 16), bf(u.z & 0xffffu), bf(u.z >> 16), bf(u.w & 0xffffu), bf(u.w >> 16)};
            #pragma unroll
            for (int e = 0; e < 8; e++) wf[h][e] = f[e];
        }
    for (int t = 0; t < T; t++) {
        float c = jc < n ? cur[(size_t)jc * cs + (size_t)t * D + i] : 0.f, s[2];
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            float v = 0.f;
            if (jc < n) {
                const float4* gg = (const float4*)(g + ((size_t)jc * T + t) * 64) + l8 * 2;
                float4 a = gg[0], b = gg[1];
                v = wf[h][0] * a.x + wf[h][1] * a.y + wf[h][2] * a.z + wf[h][3] * a.w + wf[h][4] * b.x + wf[h][5] * b.y + wf[h][6] * b.z + wf[h][7] * b.w;
            }
            for (int o = 4; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            s[h] = v;
        }
        __syncwarp();
        if (l8 == 0 && jc < n) out[((size_t)jc * T + t) * D + i] = c - tanhf(s[1]) * c + tanhf(s[0]) * msg[((size_t)jc * T + t) * D + i];
    }
}

// ---- control (T = 1 用) ----
// norm2、down の行列×ベクトル、gf の行列×ベクトル、control の4カーネルを2つにまとめ、さらに n (<= 3) 本の control を1回の起動で計算する。
// control j の入力は cur + j * cs (cs = 0 なら共通)、メッセージは msg + j * D、出力は out + j * D。足す順はまとめる前と同じ

// g[j][r] = down_j[r] . [rmsnorm(cur_j), rmsnorm(msg_j)] (r < 64)。グリッドは (8, n)。正規化の係数は cur_rms, msg_rms から読む。256 スレッド
extern "C" __global__ void ctrl_down(const float* cur, int cs, const float* msg, const bf16* w0, const bf16* w1, const bf16* w2, float* g,
                                     const float* cur_rms, const float* msg_rms) {
    PDL();
    int jc = blockIdx.y;
    cur += jc * cs;
    msg += jc * D;
    g += jc * 64;
    const bf16* w = jc == 0 ? w0 : jc == 1 ? w1 : w2;
    // 正規化の係数は、入力とメッセージを作った mix が書いたもの
    float k[2] = {cur_rms[cs ? jc : 0], msg_rms[jc]};
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= 64) return;
    const uint4* wr = (const uint4*)(w + (size_t)row * 2 * D);
    float s = 0.f;
    #pragma unroll
    for (int i = lane; i < 2 * D / 8; i += 32) {
        uint4 v = __ldg(wr + i);
        int h = i >= D / 8;
        const float* xs = (h ? msg : cur) + (i * 8 - h * D);
        float kh = k[h];
        float4 a0 = ((const float4*)xs)[0], b0 = ((const float4*)xs)[1];
        float4 a = make_float4(a0.x * kh, a0.y * kh, a0.z * kh, a0.w * kh), b = make_float4(b0.x * kh, b0.y * kh, b0.z * kh, b0.w * kh);
        s += bf(v.x & 0xffffu) * a.x + bf(v.x >> 16) * a.y + bf(v.y & 0xffffu) * a.z + bf(v.y >> 16) * a.w;
        s += bf(v.z & 0xffffu) * b.x + bf(v.z >> 16) * b.y + bf(v.w & 0xffffu) * b.z + bf(v.w >> 16) * b.w;
    }
    s = warp_sum(s);
    if (lane == 0) g[row] = s;
}

// 1ワープが要素 i について、control j の gain (gf_j の行 i) と forget (行 D + i) をレーン 8j..8j+7 で求め、out_j[i] を書く。gf は [2D][64]。
// 8 レーンで足す順は、1ワープで足していたとき (ほかのレーンは 0) と同じ。入力を全部読んでから書くので、out が cur と重なってよい
extern "C" __global__ void ctrl_up(const float* cur, int cs, float* out, const float* msg, const bf16* w0, const bf16* w1, const bf16* w2, const float* g, int n) {
    PDL();
    int i = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31, jc = lane >> 3, l8 = lane & 7;
    if (i >= D) return;
    float c = jc < n ? cur[jc * cs + i] : 0.f;
    const bf16* w = jc == 0 ? w0 : jc == 1 ? w1 : w2;
    float s[2];
    for (int h = 0; h < 2; h++) {
        float v = 0.f;
        if (jc < n) {
            uint4 u = __ldg((const uint4*)(w + (size_t)(h * D + i) * 64) + l8);
            float4 a = ((const float4*)(g + jc * 64))[l8 * 2], b = ((const float4*)(g + jc * 64))[l8 * 2 + 1];
            v += bf(u.x & 0xffffu) * a.x + bf(u.x >> 16) * a.y + bf(u.y & 0xffffu) * a.z + bf(u.y >> 16) * a.w;
            v += bf(u.z & 0xffffu) * b.x + bf(u.z >> 16) * b.y + bf(u.w & 0xffffu) * b.z + bf(u.w >> 16) * b.w;
        }
        for (int o = 4; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        s[h] = v;
    }
    if (l8 == 0 && jc < n) out[jc * D + i] = c - tanhf(s[1]) * c + tanhf(s[0]) * msg[jc * D + i];
}

// (T = 1 用) Q, K, V の3つの行列×ベクトルを1回の起動で。行 r < r0 は (w0, x0, y0)、r < r0 + r1 は (w1, x1, y1)、残りは (w2, x2, y2)。列は D
extern "C" __global__ void mv3_bf16(const bf16* w0, const float* x0, float* y0, int r0, const bf16* w1, const float* x1, float* y1, int r1,
                                    const bf16* w2, const float* x2, float* y2, int r2) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= r0 + r1 + r2) return;
    const bf16* w = w0;
    const float* x = x0;
    float* y = y0;
    if (row >= r0 + r1) w = w2, x = x2, y = y2, row -= r0 + r1;
    else if (row >= r0) w = w1, x = x1, y = y1, row -= r0;
    const uint4* wr = (const uint4*)(w + (size_t)row * D);
    for (int i = lane * 8; i < D / 8; i += 256) asm volatile("prefetch.global.L2 [%0];" :: "l"(wr + i));
    PDL();
    float s = 0.f;
    #pragma unroll 4
    for (int i = lane; i < D / 8; i += 32) {
        uint4 v = __ldg(wr + i);
        const float4* xx = (const float4*)(x + i * 8);
        float4 a = xx[0], b = xx[1];
        s += bf(v.x & 0xffffu) * a.x + bf(v.x >> 16) * a.y + bf(v.y & 0xffffu) * a.z + bf(v.y >> 16) * a.w;
        s += bf(v.z & 0xffffu) * b.x + bf(v.z >> 16) * b.y + bf(v.w & 0xffffu) * b.z + bf(v.w >> 16) * b.w;
    }
    s = warp_sum(s);
    if (lane == 0) y[row] = s;
}

// (T = 1 用) K と V の畳み込み (conv) と qkv_post を1回の起動で。ブロックは 64 個 x 128 スレッドで、qkv_post と同じく
// 0..23 が Q のヘッド、24..47 が R、48..55 が K、56..63 が V。K と V のブロックは、ヘッドの 128 チャンネルの畳み込みを計算してから正規化する。
// 畳み込みの式と足す順は conv と同じ (dyn は 1024、cph は 128)。hk, hv は K, V の hist、sk, sv は seq
extern "C" __global__ void qkv_conv_post(float* qx, const float* kx, const float* vx, const float* kw, const float* vw, int dyn,
                                         float* hk, float* hv, float* sk, float* sv, const float* qclip, const float* rclip, const int* posp,
                                         int ring, half_t* kc, half_t* vc, int host) {
    PDL();
    int b = blockIdx.x, t = threadIdx.x, pos = *posp;
    float* q = qx;
    float* r = q + R_OFF;
    float tau = ring ? 1.f : 1.f + 0.1f * logf(fmaxf(1.f, (pos + 1.f) / 8192.f));
    size_t row = (size_t)(ring ? pos % ring : pos) * 1024;
    if (b < 24) {
        float x = q[b * HD + t] * qclip[b * HD + t];
        x *= clip_scale(x, 22.627416997969522f);
        q[b * HD + t] = x * tau;
        return;
    }
    if (b < 48) {
        int h = b - 24;
        float x = t < NREL ? r[h * NREL + t] * rclip[h * NREL + t] : 0.f;
        x *= clip_scale(x, 8.f);
        if (t < NREL) r[h * NREL + t] = x * tau;
        return;
    }
    bool isk = b < 56;
    int h = isk ? b - 48 : b - 56, c = h * HD + t;
    const float* cur = isk ? kx : vx;
    const float* w = isk ? kw : vw;
    float* hist = isk ? hk : hv;
    float* seq = isk ? sk : sv;
    // conv (T = 1、ch = 1024)
    float h0 = hist[c * 3], h1 = hist[c * 3 + 1], h2 = hist[c * 3 + 2];
    seq[c] = h0;
    seq[1024 + c] = h1;
    seq[2048 + c] = h2;
    float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    float x = cur[c];
    seq[3 * 1024 + c] = x;
    float o = w0 * h0 + w1 * h1 + w2 * h2 + w3 * x;
    if (dyn >= 0) {
        const float* d = cur + dyn + h * 4;
        o += d[0] * x + d[1] * h2 + d[2] * h1 + d[3] * h0;
    }
    float y = (x + o) * (dyn >= 0 ? 1.f / 3.f : rsqrtf(5.f));
    hist[c * 3] = h1;
    hist[c * 3 + 1] = h2;
    hist[c * 3 + 2] = x;
    // qkv_post
    if (isk) kc[row + c] = f2h(y * clip_scale(y, 22.627416997969522f));
    else vc[row + c] = f2h(y * rsqrtf(block_sum(y * y) / HD + 1e-6f));
    // KV が主記憶にあれば、CPU が読む前に書き終える
    if (host) __threadfence_system();
}

// (2 <= T <= 8 用) router_topk の、T トークンをまとめて計算する版。重みの行は1回読んで T 本と積和する。
// 64 ブロック x 256 スレッドで、ブロックが 2 行、最後に終わったブロックがトークンごとに topk をとる (ワープ t がトークン t)
extern "C" __global__ void router_topk_t(const bf16* __restrict__ w, const float* __restrict__ x, float* rl, const float* bias, int* ids, float* wts,
                                         unsigned* done, int T) {
    __shared__ float part[2][4][8];
    int lane = threadIdx.x & 31, wq = (threadIdx.x >> 5) & 3, rb = threadIdx.x >> 7, row = blockIdx.x * 2 + rb;
    const uint4* wr = (const uint4*)(w + (size_t)row * D) + wq * (D / 32);
    if (lane < 9) asm volatile("prefetch.global.L2 [%0];" :: "l"(wr + lane * 8));
    PDL();
    float acc[8] = {};
    #pragma unroll
    for (int i = lane; i < D / 32; i += 32) {
        uint4 v = __ldg(wr + i);
        float wf[8] = {bf(v.x & 0xffffu), bf(v.x >> 16), bf(v.y & 0xffffu), bf(v.y >> 16), bf(v.z & 0xffffu), bf(v.z >> 16), bf(v.w & 0xffffu), bf(v.w >> 16)};
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            if (t >= T) break;
            const float4* xx = (const float4*)(x + (size_t)t * D + wq * (D / 4) + i * 8);
            float4 a = xx[0], b = xx[1];
            acc[t] += wf[0] * a.x + wf[1] * a.y + wf[2] * a.z + wf[3] * a.w + wf[4] * b.x + wf[5] * b.y + wf[6] * b.z + wf[7] * b.w;
        }
    }
    #pragma unroll
    for (int t = 0; t < 8; t++) {
        if (t >= T) break;
        float s = warp_sum(acc[t]);
        if (lane == 0) part[rb][wq][t] = s;
    }
    __syncthreads();
    if (threadIdx.x < 2 * T) {
        int r = threadIdx.x / T, t = threadIdx.x % T;
        rl[t * NEXP + blockIdx.x * 2 + r] = part[r][0][t] + part[r][1][t] + part[r][2][t] + part[r][3][t];
    }
    __shared__ bool last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(done, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (threadIdx.x == 0) *done = 0;
    int t = threadIdx.x >> 5;
    if (t < T) topk_warp(rl + t * NEXP, bias, ids + t * TOPK, wts + t * TOPK, lane);
}

// (T = 1 用) ルーターの行列×ベクトル (128 行) と topk を1回の起動で。64 ブロック x 256 スレッド。
// 各ブロックが 2 行を書き、最後に終わったブロックが topk をとる。done はブロックの数を数える (最後のブロックが 0 に戻す)
extern "C" __global__ void router_topk(const bf16* __restrict__ w, const float* __restrict__ x, float* rl, const float* bias, int* ids, float* wts,
                                       unsigned* done) {
    // 1 行を 4 ワープで分ける (ワープ q が 16 バイト単位の 72q..72q+71 番目)。ブロック (8 ワープ) は 2 行
    __shared__ float part[2][4];
    int lane = threadIdx.x & 31, wq = (threadIdx.x >> 5) & 3, rb = threadIdx.x >> 7, row = blockIdx.x * 2 + rb;
    const uint4* wr = (const uint4*)(w + (size_t)row * D) + wq * (D / 32);
    const float* xq = x + wq * (D / 4);
    if (lane < 9) asm volatile("prefetch.global.L2 [%0];" :: "l"(wr + lane * 8));
    PDL();
    float s = 0.f;
    #pragma unroll
    for (int i = lane; i < D / 32; i += 32) {
        uint4 v = __ldg(wr + i);
        const float4* xx = (const float4*)(xq + i * 8);
        float4 a = xx[0], b = xx[1];
        s += bf(v.x & 0xffffu) * a.x + bf(v.x >> 16) * a.y + bf(v.y & 0xffffu) * a.z + bf(v.y >> 16) * a.w;
        s += bf(v.z & 0xffffu) * b.x + bf(v.z >> 16) * b.y + bf(v.w & 0xffffu) * b.z + bf(v.w >> 16) * b.w;
    }
    s = warp_sum(s);
    if (lane == 0) part[rb][wq] = s;
    __syncthreads();
    if (threadIdx.x < 2) rl[blockIdx.x * 2 + threadIdx.x] = part[threadIdx.x][0] + part[threadIdx.x][1] + part[threadIdx.x][2] + part[threadIdx.x][3];
    __shared__ bool last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(done, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (threadIdx.x == 0) *done = 0;
    if (threadIdx.x < 32) topk_warp(rl, bias, ids, wts, lane);
}

// (測定用) 前のカーネルの終わりを待って、GPU の時刻 (ns) を t[i] に書く
extern "C" __global__ void stamp(unsigned long long* t, int i) {
    PDL();
    unsigned long long g;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g));
    t[i] = g;
}

// (生成) 主記憶に CPU が書いた入力 x [D] と位置を VRAM に写す。1トークン分の CUDA Graph の最初のカーネル
extern "C" __global__ void from_host(const float* xh, const int* posh, float* x, int* pos) {
    PDL();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D) x[i] = __ldcv(xh + i);
    if (i == 0) *pos = __ldcv(posh);
}

// (生成) 最大の logits の (値, 位置) を主記憶に写し、終わりの旗を立てる。CUDA Graph の最後のカーネル
extern "C" __global__ void done_host(const unsigned long long* best, unsigned long long* out, unsigned* flag) {
    PDL();
    *(volatile unsigned long long*)out = *best;
    __threadfence_system();
    *(volatile unsigned*)flag = 1;
}
