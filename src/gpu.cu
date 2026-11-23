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
    PDL();
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

// 因果的な短い畳み込みと残差。トークンを順に流す。w は [ch][4] で k=3 が現在、hist は [ch][3]
// 入力の行は cs 要素おき、出力の行は os 要素おき。dyn >= 0 なら入力の行の dyn 番目からがヘッドごとの [lag] のタップ
// seq が 0 でなければ、[直前の入力3つ; このバッチの入力 T 個] を seq [3 + T][ch] に残す。投機的デコードで、受け入れた位置までの状態に戻すため
// res が 0 でなければ、出力を res にも足す (残差)
extern "C" __global__ void conv(const float* cur, int cs, const float* w, int dyn, float* hist, float* out, int os, int ch, int cph, int T, float* seq, float* res) {
    PDL();
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
        float y = (x + o) * (dyn >= 0 ? 1.f / 3.f : rsqrtf(5.f));
        out[(size_t)t * os + c] = y;
        if (res) res[(size_t)t * os + c] += y;
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

// (prefill、VRAM の KV) 注意を計算し、出力ゲートをかけて out [T][3072] に書く。ring > 0 なら SWA 層で、直近 EXTENT 個の位置を見る
extern "C" __global__ void attn_tile(const float* qx, const half_t* kc, const half_t* vc, const float* rel_proj, float* out, const int* posp, int ring, int T) {
    PDL();
    int kvh = blockIdx.x, qi = threadIdx.x / AP, part = threadIdx.x % AP;
    int t0 = blockIdx.y * AQ, nq = min(AQ, T - t0), p0 = *posp + t0;
    __align__(16) float qr[3][AD], o[3][AD];
    float m[3], z[3];
    attn_init(qx, nullptr, 1, t0, nq, kvh, qr, o, m, z);
    attn_keys(qx, kc, vc, 0, ring, ring ? max(0, p0 + 1 - EXTENT) : 0, p0 + nq - 1, rel_proj, t0, nq, p0, kvh, qr, o, m, z);
    if (qi < nq) {
        #pragma unroll
        for (int h = 0; h < 3; h++) {
            int head = kvh * 3 + h;
            const float* gate = qx + (size_t)(t0 + qi) * QX + GATE_OFF + head * HD + part * AD;
            float* dst = out + (size_t)(t0 + qi) * 3072 + head * HD + part * AD;
            #pragma unroll
            for (int d = 0; d < AD; d++) dst[d] = o[h][d] / z[h] * sigmoidf(gate[d]);
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
    PDL();
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= rows) return;
    const uint4* wr = (const uint4*)(w + (size_t)row * cols);
    float acc[8] = {};
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

// (prefill 用) T*8 個の割り当て (トークン t の k 番目 = t*8+k) を expert 順に並べる。order[off[e]..off[e+1]] が expert e の割り当て
extern "C" __global__ void moe_sort(const int* ids, int n, int* order, int* off) {
    PDL();
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

// y[t][r] = scale[r] * sum_c W[r][c] x[t][c] (t < T <= 8)。W は FP8 の [rows, cols]、cols は 16 の倍数。1行を1ワープで計算する
extern "C" __global__ void mv_fp8(const u8* __restrict__ w, const float* __restrict__ sc, const float* __restrict__ x, float* __restrict__ y,
                                  int rows, int cols, int T) {
    PDL();
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= rows) return;
    const uint4* wr = (const uint4*)(w + (size_t)row * cols);
    float acc[8] = {};
    for (int i = lane; i < cols / 16; i += 32) {
        uint4 v = __ldg(wr + i);
        unsigned u[4] = {v.x, v.y, v.z, v.w};
        float wf[16];
        #pragma unroll
        for (int k = 0; k < 16; k++) wf[k] = e4m3(u[k / 4] >> (8 * (k % 4)));
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            if (t >= T) break;
            const float4* xx = (const float4*)(x + (size_t)t * cols + i * 16);
            float s = 0.f;
            #pragma unroll
            for (int q = 0; q < 4; q++) {
                float4 a = xx[q];
                s += wf[4 * q] * a.x + wf[4 * q + 1] * a.y + wf[4 * q + 2] * a.z + wf[4 * q + 3] * a.w;
            }
            acc[t] += s;
        }
    }
    #pragma unroll
    for (int t = 0; t < 8; t++) {
        if (t >= T) break;
        float s = warp_sum(acc[t]);
        if (lane == 0) y[(size_t)t * rows + row] = s * sc[row];
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

// ブロック 8 行の注意。キーは文脈の位置 lo..hi (キャッシュ) とブロックの 8 行で、マスクなし。グリッドは (ヘッド 24, 8)、128スレッド
extern "C" __global__ void dr_attn(const float* qkv, const half_t* kc, const half_t* vc, int ring, int lo, int hi, float* out) {
    PDL();
    __shared__ float sc[DRAFT_CTX + 8];
    int h = blockIdx.x, t = blockIdx.y, kvh = h / 3, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int nc = hi - lo, n = nc + 8;
    float4 qv = ((const float4*)(qkv + (size_t)t * DQKV + h * HD))[lane];
    for (int j = warp; j < n; j += blockDim.x >> 5) {
        float4 kv = j < nc ? h4(kc + kv_row(lo + j, ring) + kvh * HD + lane * 4)
                           : ((const float4*)(qkv + (size_t)(j - nc) * DQKV + 3072 + kvh * HD))[lane];
        float s = warp_sum(qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w);
        if (lane == 0) sc[j] = s * rsqrtf((float)HD);
    }
    __syncthreads();
    float m = -INFINITY;
    for (int j = tid; j < n; j += blockDim.x) m = fmaxf(m, sc[j]);
    m = block_max(m);
    float z = 0.f;
    for (int j = tid; j < n; j += blockDim.x) z += sc[j] = expf(sc[j] - m);
    z = block_sum(z);
    float o = 0.f;
    for (int j = 0; j < n; j++)
        o += sc[j] * (j < nc ? h2f(vc[kv_row(lo + j, ring) + kvh * HD + tid]) : qkv[(size_t)(j - nc) * DQKV + 4096 + kvh * HD + tid]);
    out[(size_t)t * 3072 + h * HD + tid] = o / z;
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

// Markov ヘッドの1手: logits = softcap(l + fb + base)、その最大の位置と sum exp(logits - max)。1ブロック 1024スレッド
extern "C" __global__ void fold_finish(float* l, const float* fb, const float* base, int n, int* best, float* sumexp) {
    PDL();
    __shared__ int bi[32];
    float v = -INFINITY;
    int i = 0;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        float x = 30.f * tanhf((l[j] + fb[j] + base[j]) / 30.f);
        l[j] = x;
        if (x > v) v = x, i = j;
    }
    for (int o = 16; o > 0; o >>= 1) {
        float v2 = __shfl_xor_sync(0xffffffffu, v, o);
        int i2 = __shfl_xor_sync(0xffffffffu, i, o);
        if (v2 > v || (v2 == v && i2 < i)) v = v2, i = i2;
    }
    float m = block_max(v);
    if ((threadIdx.x & 31) == 0) bi[threadIdx.x >> 5] = v == m ? i : 0x7fffffff;
    __syncthreads();
    float z = 0.f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) z += expf(l[j] - m);
    z = block_sum(z);
    if (threadIdx.x == 0) {
        int b = 0x7fffffff;
        for (int w = 0; w < (int)(blockDim.x >> 5); w++) b = min(b, bi[w]);
        *best = b;
        *sumexp = z;
    }
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
