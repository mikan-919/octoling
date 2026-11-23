//! GPU (CUDA) 版の順伝播。計算の順序は model.rs の CPU 版と同じ。
//! 20〜23層の MXFP8 expert は VRAM に載らないので CPU で計算する。埋め込み表も CPU に置く。
//! 生成 (forward) は1トークンずつで、カーネル列を CUDA Graph に記録して再生する。
//! prefill は最大 TMAX トークンをまとめて処理し、重みを読む回数をトークン数で割る。

use crate::model::{self, BLOCK, CORE, D, EXTENT, FF, GLOBAL_LAYERS, Layer, MXFP8_FROM, Model, N_EXP, N_LAYER, N_SLOT, PASSES, TOP_K, VOCAB, W};
use cudarc::driver::{CudaContext, CudaEvent, CudaFunction, CudaGraph, CudaModule, CudaStream, result, sys};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;
use std::time::Instant;

/// global 層の KV キャッシュを VRAM に置ける文脈の長さ。KV は f16 で、8192 トークンで global 層 268 MB、SWA 層のリング 250 MB。
/// 文脈長 (Gpu::new の ctx) がこれより長いと、global 層の KV を GPU から直接読める主記憶 (ページ固定) に置く。
/// 1 位置あたり 8 スロット x (K + V) x 2 KB = 32 KB で、256K トークンなら 8.6 GB。
/// ponytail: 主記憶の KV は f16 のまま。8 ビットにすれば量も生成時の読み出しも半分になる
pub const VRAM_CTX: usize = 8192;

/// prefill で一度に処理するトークン数の上限
pub const TMAX: usize = 512;

/// SWA 層の KV のリングバッファの行数。バッチの全トークンを書いてから注意を計算するので、窓幅に TMAX を足す
const RING: usize = EXTENT + TMAX;

/// 投機的デコードで一度に検証するトークン数の上限
pub const SPEC_T: usize = 8;

/// 下書きモデルが読む層出力のスロット (層の適用の 4, 22, 40 番目)
const TAP_SLOTS: [usize; 3] = [4, 28, 46];

type P = u64; // デバイスのポインタ

trait Arg {
    fn bits(self) -> u64;
}
impl Arg for u64 {
    fn bits(self) -> u64 {
        self
    }
}
impl Arg for i32 {
    fn bits(self) -> u64 {
        self as u32 as u64
    }
}
impl Arg for f32 {
    fn bits(self) -> u64 {
        self.to_bits() as u64
    }
}

trait Grid {
    fn dims(self) -> (u32, u32);
}
impl Grid for usize {
    fn dims(self) -> (u32, u32) {
        (self as u32, 1)
    }
}
impl Grid for i32 {
    fn dims(self) -> (u32, u32) {
        (self as u32, 1)
    }
}
impl Grid for (usize, usize) {
    fn dims(self) -> (u32, u32) {
        (self.0 as u32, self.1 as u32)
    }
}

macro_rules! k {
    ($g:expr, $name:literal, $grid:expr, $block:expr, $smem:expr; $($a:expr),* $(,)?) => {
        $g.launch($name, Grid::dims($grid), $block as u32, $smem as u32, &[$(Arg::bits($a)),*])
    };
}

struct GCtrl {
    down: P,
    gf: P, // [gain; forget] を縦に並べた [2D, RANK]
}

struct GLayer {
    global: bool,
    norms: P, // [3][D]。SWA は attn_norm 1本
    attn_query: P,
    mlp_query: P,
    msg_query: P,
    wq: P, // [wq; wr; gate_attn; sink]
    wk: P, // [wk; k_dyn]
    wv: P, // [wv; v_dyn]
    wo: P,
    q_clip: P,
    rel_clip: P,
    rel_proj: P,
    k_conv: P,
    v_conv: P,
    attn_conv: P,
    mlp_conv: P,
    mlp_norm: P,
    router: P,
    router_bias: P,
    ctrl: Vec<GCtrl>,
    experts: GExperts,
}

#[derive(Clone, Copy)]
enum GExperts {
    Fp4([P; 6]),            // w13, s13, a13, w2, s2, a2
    Sk { tab: P, ixd: P },  // SQ 形式。tab は行列ごとの SkMat、ixd は活性の逆スケール [128][FF]
    Host,                   // MXFP8。CPU と分担する
}

struct GSlot {
    kc: P,
    vc: P,
    host: Option<[usize; 2]>, // KV が主記憶にあれば、その K と V の主記憶側のアドレス
    hist: [P; 4], // k, v, attn 出力, mlp 出力
    seq: [P; 4],  // 検証のバッチの [直前の入力3つ; 入力 SPEC_T 個]。外れた下書きを捨てるとき hist を戻す
}

// cap トークン分の作業領域。並びは gpu.cu の先頭に書いたとおり [種類][T][要素]。
// 生成と検証 (T <= SPEC_T) 用の小さい組と、prefill 用の大きい組がある。大きい組は expert の枠と同じ場所に置き、prefill の間は枠を空にする
struct Bufs {
    cap: usize,
    x: P,
    pos: P,    // i32。バッチの先頭トークンの位置。カーネルはここから読む
    blocks: P, // [8][T][D]
    prelude: P,
    bank: P, // [2][T][D]
    msgs: P, // [4][T][D]。T <= SPEC_T なら、ループの 12 層分を周の始めにまとめて作るので [12][4][T][D]
    inp: P,  // [3][T][D]
    qx: P,
    kx: P,
    vx: P,
    kp: P,
    vp: P,
    o: P,
    a: P,
    h: P,
    y: P,
    cat: P,
    g: P,
    gf: P,
    rl: P,
    ids: P,
    wts: P,
    act: P, // [T*8][FF]
    gu: P,  // [T*8][2FF]
    yd: P,  // [T*8][D]
    order: P,
    off: P,
    part: P, // [24][NSPLIT][HD + 2]。生成時の注意の部分結果
    logits: P,
    taps: P,       // [4][cap][D]。下書きモデルが読む、スロット 4, 28, 46 の層出力と最後の正規化の出力
    amax: P,       // [SPEC_T] i32。検証のバッチの各行の、最大の logits の位置
    logits_all: P, // [SPEC_T][201024]
    acc: P,        // [cap][24][HD + 2]。主記憶の KV の塊ごとに足し込む注意の途中結果
    done: P,       // u32 [SPEC_T * 24 + 1]。最後に終わったブロックを決める数 (attn_part のヘッドごと、router_topk)。使い終わると 0 に戻る
    best: P,       // u64。最後の行の最大の logits の (値, 位置) を並べたもの (softcap が書く)
    route: P,      // 生成の 20〜23層で、GPU で計算する枠 (CPU との受け渡しの route を写したもの、u32 x 12)
    rms: P,        // (T = 1) RMS の逆数: [0..3] 注意の入力、[3] FFN の入力、[4..52] ループ 12 層のメッセージ
}

const QX: usize = 3072 + 384 + 3072 + 24;

impl Bufs {
    /// t トークン分をとる。alloc(バイト数) が場所を返す。acc は主記憶の KV 用の途中結果をとるか
    fn new(t: usize, acc: bool, mut alloc: impl FnMut(usize) -> P) -> Self {
        let mut fz = |n: usize| alloc(n * 4);
        Bufs {
            cap: t,
            x: fz(t * D),
            pos: fz(1),
            blocks: fz(8 * t * D),
            prelude: fz(t * D),
            bank: fz(2 * t * D),
            msgs: fz(if t <= SPEC_T { (CORE.1 - CORE.0) * 4 * t * D } else { 4 * t * D }),
            inp: fz(3 * t * D),
            qx: fz(t * QX),
            kx: fz(t * (1024 + 32)),
            vx: fz(t * (1024 + 32)),
            kp: fz(t * 1024),
            vp: fz(t * 1024),
            o: fz(t * 3072),
            a: fz(t * D),
            h: fz(t * D),
            y: fz(t * D),
            cat: fz(t * 2 * D),
            g: fz(t * 64),
            gf: fz(t * 2 * D),
            rl: fz(t * 128),
            ids: fz(t * TOP_K),
            wts: fz(t * TOP_K),
            act: fz(t * TOP_K * FF),
            gu: fz(t * TOP_K * 2 * FF),
            yd: fz(t * TOP_K * D),
            order: fz(t * TOP_K),
            off: fz(129),
            logits: fz(201024),
            part: fz(SPEC_T * 24 * 32 * 130),
            taps: fz(4 * t * D),
            amax: fz(SPEC_T),
            logits_all: fz(SPEC_T * 201024),
            acc: if acc { fz(t * 24 * 130) } else { 0 },
            done: fz(SPEC_T * 24 + 1),
            best: fz(2),
            route: fz(16),
            rms: fz(64),
        }
    }
}

pub struct Gpu {
    cpu: &'static Model,
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    f: HashMap<&'static str, CudaFunction>,
    layers: Vec<GLayer>,
    slots: Vec<GSlot>,
    unembed: P,
    norm: P,
    output_query: P,
    msg_queries: P, // ループの 12 層の msg_query を並べたもの [12][4][D]
    small: Bufs,
    big: Bufs, // expert の枠と同じ場所
    pos: usize,
    ctx: usize,
    // T トークン分の GPU の処理を、CPU で MoE を計算する所で区切って記録した CUDA Graph。(T, 検証か) ごと
    graphs: RefCell<HashMap<(usize, bool), Vec<CudaGraph>>>,
    recording: RefCell<Vec<CudaGraph>>,
    rewind_tab: P, // conv_rewind に渡す (seq, hist, ch) の表
    capturing: Cell<bool>,
    stamps: RefCell<Vec<&'static str>>, // (OCTO_STAMP) 時刻を書いた順のカーネル名
    stamp_buf: P,
    // prefill で、20〜23層の MXFP8 expert の一部を PCIe で送って GPU で計算するための送信用ストリームと置き場
    copy: Arc<CudaStream>,
    staging: [P; 2],
    up_done: [CudaEvent; 2],
    comp_done: [CudaEvent; 2],
    xc: RefCell<XCache>,
    fill_done: CudaEvent, // 枠への転送の終わり
    // 生成の 20〜23層の MoE で CPU と受け渡す、GPU から直接読み書きできる主記憶 (GPU 側, 主記憶側のアドレス)。層ごとに HX バイト
    hx: (P, usize),
    // 生成の入力 (x, 位置) と結果 (旗, 最大の位置, logits) を受け渡す主記憶。HIO_* の並び。転送エンジンを使わないので、枠への大きい転送の後ろで待たされない
    hio: (P, usize),
    // 生成で、主記憶の KV への注意を CPU と受け渡す主記憶 (GPU 側, 主記憶側のアドレス)。スロットごとに HA バイト。文脈が VRAM_CTX 以下なら使わない
    ha: (P, usize),
    host_seq: RefCell<Vec<HostEv>>, // 1トークンのグラフが CPU を待つ順番
    token_done: CudaEvent, // 生成の1トークン分の CUDA Graph の終わり
}

/// 生成のグラフが CPU に頼む処理
#[derive(Clone, Copy)]
enum HostEv {
    Attn(usize), // スロット番号
    Moe(usize),  // 20 層からの番号
}

/// prefill で1回に PCIe で送る MXFP8 expert の数。置き場は2つあり、片方を送るあいだにもう片方で計算する
const GROUP: usize = 8;
// MXFP8 の expert 1個あたりのバイト数
const W13: usize = 2 * FF * D;
const S13: usize = W13 / 32;
const W2: usize = D * FF;
const S2: usize = W2 / 32;
/// expert 1個分の枠 [w13 | s13 | w2 | s2] のバイト数
const XSLOT: usize = W13 + S13 + W2 + S2;
/// 枠の使用回数を、1トークンごとにこの割合に減らす。直近によく選ばれた expert を残す
const XDECAY: f32 = 0.95;
/// 1層の計算のあとに入れ替える枠の数の上限 (1つ 8.2 MB を PCIe で送る)
const XFILLS: usize = 2;
/// 枠を確保したあとに残す VRAM。Windows (WDDM) は VRAM が足りなくなると主記憶へ退避させ、何倍も遅くなる
const VRAM_MARGIN: usize = 150 << 20;

// CPU との受け渡しの場所の並び (層ごと、バイト)。h と y は f32 x D、route は [n u32, 枠の番号 u64 (+8), 重み f32 x 8 (+16)]、旗は u32
const HX: usize = 20480;
const HX_H: usize = 0;
const HX_Y: usize = 4 * D;
const HX_ROUTE: usize = 8 * D;
const HX_FH: usize = 8 * D + 128; // h を書いた (GPU が立てる)
const HX_FR: usize = 8 * D + 192; // route を書いた (CPU が立てる)
const HX_FY: usize = 8 * D + 256; // y を書いた (CPU が立てる)
// 主記憶の KV への注意の受け渡し。[q 3072, r 384, sink 24] の f32、結果 [3072] の f32、旗
const HA: usize = 32768;
const HA_Q: usize = 0;
const HA_O: usize = 16384;
const HA_FQ: usize = 28672; // q, r, sink を書いた (GPU が立てる)
const HA_FO: usize = 28736; // 結果を書いた (CPU が立てる)
// 生成の入出力の受け渡しの場所の並び (バイト)
const HIO_X: usize = 0;
const HIO_POS: usize = 4 * D;
const HIO_DONE: usize = 4 * D + 64; // トークンの計算が終わった (GPU が立てる)
const HIO_BEST: usize = 4 * D + 128;
const HIO_LOGITS: usize = 16384;

/// 生成で、20〜23層 (MXFP8) のよく選ばれる expert を VRAM に置いておく枠。選ばれた expert が枠にあれば GPU で、なければ CPU で計算する。
/// 枠 j は base + j * XSLOT。先頭の 2 * GROUP 枠は prefill の置き場 (staging) と同じ場所なので、prefill で使ったら空にする
struct XCache {
    base: P,
    reserved: usize, // 先頭のこの数の枠は、prefill の置き場と作業領域を兼ねる
    owner: Vec<Option<(usize, usize)>>, // 枠 → (層 - 20, expert)
    slot: [[Option<usize>; N_EXP]; 4],
    cnt: [[f32; N_EXP]; 4], // 選ばれた回数 (XDECAY で減らす)
    pending: Vec<(usize, usize, usize)>, // このトークンで入れ替えると決めた (層 - 20, expert, 枠)。23層のあとにまとめて送る
    ready: Vec<bool>,   // 枠の中身が届いているか。入れ替えると決めた枠は、転送が終わるまで使わない
    issued: Vec<usize>, // 転送を出したが、届いたのをまだ確かめていない枠
}

/// GPU と CPU の分担を釣り合わせるための見積もり (秒)。expert 8個 (66 MB) を PCIe で送る時間と、CPU で割り当て1つを計算する時間。
/// 実測: PCIe 4.0 x8 で 14.3 GB/s、CPU は 512 トークンの1層 (4096 割り当て) で約 130 ms。別の機械では測り直す
const GPU_PER_GROUP: f64 = 4.6e-3;
const CPU_PER_ASSIGN: f64 = 32e-6;

const KERNELS: [&str; 58] = [
    "mv_bf16", "gemm_bf16", "rmsnorm", "mix", "norm2", "control", "conv", "qkv_post", "attn_part", "attn_tile", "add", "softcap", "topk", "moe_up_nvfp4", "moe_down_nvfp4",
    "moe_sort", "gemm_nvfp4_grouped", "situ_rows", "moe_sum", "gemm_mxfp8_grouped", "situ_sel", "conv_rewind", "argmax_rows",
    "quant_rows_bf16", "quant_cols_f32", "mv_fp8", "tapmix", "dr_ctx_store", "dr_qk", "dr_attn", "silu_mul", "concat_frnn", "add_bias_rows", "fold_finish", "mv_bf16_t", "attn_blk", "attn_fin", "moe_up_mx", "moe_down_mx", "ctrl_down", "ctrl_up", "mv3_bf16", "qkv_conv_post", "router_topk", "stamp", "to_host", "wait_host", "add_host", "from_host", "done_host", "attn_to_host", "attn_from_host", "moe_up_mx_g", "moe_down_mx_g", "moe_up_sk", "moe_down_sk", "gemm_sk_grouped", "situ_sk",
];

fn as_bytes<T>(v: &[T]) -> &[u8] {
    // SAFETY: u16/f32/u8 の配列をバイト列として読むだけ
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}

fn f4(n: usize) -> u64 {
    (n * 4) as u64
}

impl Gpu {
    /// n_ctx は文脈長の上限。VRAM_CTX より長ければ、global 層の KV を主記憶に置く
    pub fn new(mut cpu: Model, n_ctx: usize) -> Self {
        let ctx = CudaContext::new(0).expect("GPU を開けない");
        ctx.bind_to_thread().unwrap();
        // 20〜23層の MXFP8 expert (4.2 GB) をページ固定メモリに写す。prefill で PCIe から非同期に送るため。
        // CPU の計算もこちらを読むので、mmap した元のページは OS が追い出せる
        for l in cpu.layers.iter_mut().filter(|l| l.experts.mx) {
            let e = &mut l.experts;
            (e.w13, e.s13, e.w2, e.s2) = (pin(e.w13), pin(e.s13), pin(e.w2), pin(e.s2));
        }
        let cpu: &'static Model = Box::leak(Box::new(cpu));
        let ptx = compile_ptx_with_opts(include_str!("gpu.cu"),CompileOptions { arch: Some("sm_120"), ..Default::default() })
            .unwrap_or_else(|e| panic!("CUDA カーネルのコンパイルに失敗: {e}"));
        // 既定のストリームは CUDA Graph に記録できない
        let stream = ctx.new_stream().unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let f = KERNELS.iter().map(|&n| (n, module.load_function(n).unwrap())).collect();

        let mut up = Uploader::new(&stream);
        let f32s = |v: &[f32], up: &mut Uploader| up.bytes(as_bytes(v));
        let bfs = |parts: &[W], up: &mut Uploader| up.bytes(as_bytes(&parts.concat()));
        let flat = |c: &[[f32; 4]]| c.iter().flatten().copied().collect::<Vec<f32>>();
        let layers = cpu
            .layers
            .iter()
            .map(|l: &Layer| {
                let norms: Vec<f32> = if l.global { l.norms.iter().flat_map(|&w| model::bf_vec(w)).collect() } else { model::bf_vec(l.norms[0]) };
                let wk: Vec<W> = [Some(l.wk), l.k_dyn].into_iter().flatten().collect();
                let wv: Vec<W> = [Some(l.wv), l.v_dyn].into_iter().flatten().collect();
                let e = &l.experts;
                GLayer {
                    global: l.global,
                    norms: f32s(&norms, &mut up),
                    attn_query: l.attn_query.map_or(0, |q| f32s(q, &mut up)),
                    mlp_query: f32s(l.mlp_query, &mut up),
                    msg_query: l.msg_query.as_ref().map_or(0, |q| f32s(q, &mut up)),
                    wq: bfs(&[l.wq, l.wr, l.gate_attn, l.sink], &mut up),
                    wk: bfs(&wk, &mut up),
                    wv: bfs(&wv, &mut up),
                    wo: up.bytes(as_bytes(l.wo)),
                    q_clip: f32s(&model::bf_vec(l.q_clip), &mut up),
                    rel_clip: f32s(&model::bf_vec(l.rel_clip), &mut up),
                    rel_proj: f32s(&model::bf_vec(l.rel_proj), &mut up),
                    k_conv: f32s(&flat(&l.k_conv), &mut up),
                    v_conv: f32s(&flat(&l.v_conv), &mut up),
                    attn_conv: f32s(&flat(&l.attn_conv), &mut up),
                    mlp_conv: f32s(&flat(&l.mlp_conv), &mut up),
                    mlp_norm: f32s(&model::bf_vec(l.mlp_norm), &mut up),
                    router: up.bytes(as_bytes(l.router)),
                    router_bias: f32s(l.router_bias, &mut up),
                    ctrl: l.ctrl.iter().flatten().map(|c| GCtrl { down: up.bytes(as_bytes(c.down)), gf: bfs(&[c.gain, c.forget], &mut up) }).collect(),
                    experts: if let Some(sk) = &e.sk {
                        let im = sk.image();
                        let base = up.bytes(&im.bytes);
                        let tab: Vec<[u64; 3]> = im.tab.iter().map(|m| [base + m[0], base + m[1], m[2]]).collect();
                        GExperts::Sk { tab: up.bytes(as_bytes(&tab)), ixd: f32s(&im.ixd, &mut up) }
                    } else if e.mx {
                        GExperts::Host
                    } else {
                        GExperts::Fp4([e.w13, e.s13, as_bytes(e.a13), e.w2, e.s2, as_bytes(e.a2)].map(|b| up.bytes(b)))
                    },
                }
            })
            .collect();
        let slots: Vec<GSlot> = (0..N_SLOT)
            .map(|s| {
                let rows = if GLOBAL_LAYERS.contains(&phys(s)) { n_ctx } else { RING };
                let ([kc, vc], host) = if rows > VRAM_CTX {
                    let [(kd, kh), (vd, vh)] = [0; 2].map(|_| mapped(rows * 1024 * 2));
                    ([kd, vd], Some([kh, vh]))
                } else {
                    ([0; 2].map(|_| up.zeros(rows * 1024 * 2)), None)
                };
                GSlot {
                    kc,
                    vc,
                    host,
                    hist: [1024, 1024, D, D].map(|ch| up.zeros(ch * 3 * 4)),
                    seq: [1024, 1024, D, D].map(|ch| up.zeros(ch * (SPEC_T + 3) * 4)),
                }
            })
            .collect();
        let small = Bufs::new(SPEC_T, false, |n| up.zeros(n));
        let unembed = up.bytes(as_bytes(cpu.unembed));
        let norm = f32s(&model::bf_vec(cpu.norm), &mut up);
        let output_query = f32s(cpu.output_query, &mut up);
        let mq: Vec<f32> = cpu.layers[CORE.0..CORE.1].iter().flat_map(|l| l.msg_query.as_ref().unwrap().iter().copied()).collect();
        let msg_queries = f32s(&mq, &mut up);
        // 残りの VRAM で expert の枠をとる。OCTO_EXPERT_SLOTS で数を決められる。
        // 先頭の枠は prefill の置き場 (staging、2 * GROUP 枠) と prefill 用の作業領域を兼ねる
        let align = |n: usize| n.div_ceil(256) * 256;
        let mut big_bytes = 0;
        Bufs::new(TMAX, n_ctx > VRAM_CTX, |n| {
            big_bytes += align(n);
            0
        });
        let reserved = (2 * GROUP * XSLOT + big_bytes).div_ceil(XSLOT);
        let (free, _) = result::mem_get_info().unwrap();
        // MXFP8 の層がなければ (SQ 形式) 枠は要らず、prefill の作業領域の分だけとる
        let n_slots = if cpu.layers.iter().any(|l| l.experts.mx) {
            std::env::var("OCTO_EXPERT_SLOTS")
                .map_or(free.saturating_sub(VRAM_MARGIN) / XSLOT, |s| s.parse().expect("OCTO_EXPERT_SLOTS は数"))
                .clamp(reserved, 255)
        } else {
            reserved
        };
        let xbase = up.zeros(n_slots * XSLOT);
        let staging = [xbase, xbase + (GROUP * XSLOT) as u64];
        let mut at = xbase + (2 * GROUP * XSLOT) as u64;
        let big = Bufs::new(TMAX, n_ctx > VRAM_CTX, |n| {
            let p = at;
            at += align(n) as u64;
            p
        });
        let tab: Vec<u64> = slots.iter().flat_map(|s: &GSlot| (0..4).flat_map(move |k| [s.seq[k], s.hist[k], [1024, 1024, D, D][k] as u64])).collect();
        let rewind_tab = up.bytes(as_bytes(&tab));
        let (free, _) = result::mem_get_info().unwrap();
        eprintln!("GPU に {:.2} GB 載せた (expert の枠 {n_slots} 個)。残り {:.2} GB", up.total as f64 / 1e9, free as f64 / 1e9);
        let ev = || ctx.new_event(None).unwrap();
        Self {
            cpu,
            copy: ctx.new_stream().unwrap(),
            staging,
            up_done: [ev(), ev()],
            comp_done: [ev(), ev()],
            xc: RefCell::new(XCache { base: xbase, reserved, owner: vec![None; n_slots], slot: [[None; N_EXP]; 4], cnt: [[0.0; N_EXP]; 4], pending: vec![], ready: vec![false; n_slots], issued: vec![] }),
            fill_done: ev(),
            hx: mapped(4 * HX),
            hio: mapped(HIO_LOGITS + VOCAB * 4),
            ha: if n_ctx > VRAM_CTX { mapped(N_SLOT * HA) } else { (0, 0) },
            host_seq: RefCell::new(vec![]),
            token_done: ev(),
            _ctx: ctx.clone(),
            stream,
            _module: module,
            f,
            layers,
            slots,
            unembed,
            norm,
            output_query,
            msg_queries,
            small,
            big,
            pos: 0,
            ctx: n_ctx,
            graphs: RefCell::new(HashMap::new()),
            recording: RefCell::new(vec![]),
            rewind_tab,
            capturing: Cell::new(false),
            stamps: RefCell::new(vec![]),
            stamp_buf: up.zeros(8 << 16),
        }
    }

    fn launch(&self, name: &str, grid: (u32, u32), block: u32, smem: u32, args: &[u64]) {
        let mut vals = args.to_vec();
        let mut ptrs: Vec<*mut c_void> = vals.iter_mut().map(|v| v as *mut u64 as *mut c_void).collect();
        // どのカーネルも先頭で PDL() を呼ぶので、前のカーネルが終わる前に起動してよい
        let mut attr = sys::CUlaunchAttribute {
            id: sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
            pad: [0; 4],
            value: sys::CUlaunchAttributeValue { programmaticStreamSerializationAllowed: 1 },
        };
        let cfg = sys::CUlaunchConfig {
            gridDimX: grid.0,
            gridDimY: grid.1,
            gridDimZ: 1,
            blockDimX: block,
            blockDimY: 1,
            blockDimZ: 1,
            sharedMemBytes: smem,
            hStream: self.stream.cu_stream(),
            attrs: &mut attr,
            numAttrs: 1,
        };
        // SAFETY: 引数の数と型は gpu.cu の宣言に合わせてある。ポインタはどれも 64bit
        unsafe { sys::cuLaunchKernelEx(&cfg, self.f[name].cu_function(), ptrs.as_mut_ptr(), std::ptr::null_mut()).result() }
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        if name != "stamp" && self.capturing.get() && stamping() {
            self.stamp(KERNELS.iter().find(|k| **k == name).unwrap());
        }
    }

    fn stamp(&self, name: &'static str) {
        let i = self.stamps.borrow().len();
        self.stamps.borrow_mut().push(name);
        k!(self, "stamp", 1, 32, 0; self.stamp_buf, i as i32);
    }

    fn copy(&self, dst: P, src: P, floats: usize) {
        // SAFETY: どちらも確保済みの領域
        unsafe { result::memcpy_dtod_async(dst, src, floats * 4, self.stream.cu_stream()) }.unwrap();
    }

    fn htod<T>(&self, dst: P, src: &[T]) {
        // SAFETY: dst は src 以上の長さで確保してある。転送が終わるまで待つ
        unsafe { result::memcpy_htod_async(dst, src, self.stream.cu_stream()) }.unwrap();
        self.stream.synchronize().unwrap();
    }

    fn dtoh<T: Default + Clone>(&self, src: P, n: usize) -> Vec<T> {
        let mut v = vec![T::default(); n];
        // SAFETY: src は n 要素以上ある。転送が終わるまで待つ
        unsafe { result::memcpy_dtoh_async(&mut v, src, self.stream.cu_stream()) }.unwrap();
        self.stream.synchronize().unwrap();
        v
    }

    /// Y[t] = W X[t]。T = 1 は行列×ベクトル、T <= SPEC_T は重みを1回読んで T 本と積和、それより多ければテンソルコアの行列積
    /// (テンソルコアの行列積は大きいバッチ向けで、数トークンだと読み込みの待ちで遅い)
    fn gemm(&self, w: P, x: P, y: P, rows: usize, cols: usize, t: usize) {
        if t == 1 {
            k!(self, "mv_bf16", (rows * 32).div_ceil(256), 256, 0; w, x, y, rows as i32, cols as i32);
        } else if t <= SPEC_T {
            k!(self, "mv_bf16_t", (rows * 32).div_ceil(256), 256, 0; w, x, y, rows as i32, cols as i32, t as i32);
        } else {
            k!(self, "gemm_bf16", (rows.div_ceil(64), t.div_ceil(32)), 128, 0; w, x, y, rows as i32, cols as i32, t as i32);
        }
    }

    #[allow(clippy::too_many_arguments)]
    /// norm が (重み, query ごとのずれ) なら、続けて rmsnorm をかける
    /// rms が 0 でなければ、出力の RMS の逆数を rms [nq][T] に書く
    fn mix(&self, blocks: P, nb: usize, cur: P, queries: P, nq: usize, out: P, t: usize, norm: Option<(P, usize)>, rms: P) {
        assert!(nb < 5, "mix は状態 5 個まで (MIX_NS)");
        let (nw, nws) = norm.unwrap_or((0, 0));
        k!(self, "mix", (nq, t), 768, 0; blocks, nb as i32, cur, queries, out, t as i32, nw, nws as i32, rms);
    }

    /// control cs[j] (j < 3) を、入力 j に メッセージ msg + slab(j) でかけて out + slab(j) に書く。並びは [j][T][D]。
    /// 入力 j は shared なら cur (全部共通)、そうでなければ cur + slab(j) で、out と同じでよい。T = 1 なら全部を1組の起動で計算する
    /// (T = 1) cur_rms, msg_rms は入力とメッセージの RMS の逆数 (mix が書いたもの)
    #[allow(clippy::too_many_arguments)]
    fn control(&self, cur: P, shared: bool, out: P, msg: P, cs: &[GCtrl], t: usize, cur_rms: P, msg_rms: P) {
        let b = self.bufs(t);
        let slab = |j: usize| f4(j * t * D);
        if t == 1 {
            let w = |j: usize, f: fn(&GCtrl) -> P| cs.get(j).map_or(0, f);
            let stride = if shared { 0 } else { D as i32 };
            k!(self, "ctrl_down", (8, cs.len()), 256, 0; cur, stride, msg, w(0, |c| c.down), w(1, |c| c.down), w(2, |c| c.down), b.g, cur_rms, msg_rms);
            k!(self, "ctrl_up", (D * 32).div_ceil(256), 256, 0;
                cur, stride, out, msg, w(0, |c| c.gf), w(1, |c| c.gf), w(2, |c| c.gf), b.g, cs.len() as i32);
            return;
        }
        // 共通の入力を上書きする前に写す
        for j in (0..cs.len()).rev() {
            let src = if shared { cur } else { cur + slab(j) };
            if src != out + slab(j) {
                self.copy(out + slab(j), src, t * D);
            }
        }
        for (j, c) in cs.iter().enumerate() {
            let (o, m) = (out + slab(j), msg + slab(j));
            k!(self, "norm2", (2, t), 256, 0; o, m, b.cat);
            self.gemm(c.down, b.cat, b.g, 64, 2 * D, t);
            self.gemm(c.gf, b.g, b.gf, 2 * D, 64, t);
            k!(self, "control", (D.div_ceil(256), t), 256, 0; o, m, b.gf);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn conv(&self, cur: P, cs: usize, w: P, dyn_: Option<usize>, hist: P, out: P, os: usize, ch: usize, cph: usize, t: usize, seq: P, res: P) {
        let d = dyn_.map_or(-1, |d| d as i32);
        k!(self, "conv", ch.div_ceil(256), 256, 0; cur, cs as i32, w, d, hist, out, os as i32, ch as i32, cph as i32, t as i32, seq, res);
    }

    /// トークンの埋め込みと先頭の位置を GPU に送る
    fn inputs(&self, tokens: &[u32]) {
        assert!(self.pos + tokens.len() <= self.ctx, "文脈長の上限 {} を超えた", self.ctx);
        let e: Vec<f32> = tokens
            .iter()
            .flat_map(|&tok| {
                let mut e = model::bf_row(self.cpu.embed, tok as usize, D);
                model::rmsnorm(&mut e, Some(self.cpu.embed_norm));
                e
            })
            .collect();
        if self.host_io(tokens.len()) {
            // SAFETY: hio は主記憶の受け渡しの場所。GPU はグラフの最初のカーネル (from_host) で読む
            unsafe {
                let p = self.hio.1;
                std::ptr::copy_nonoverlapping(e.as_ptr(), (p + HIO_X) as *mut f32, D);
                std::ptr::write_volatile((p + HIO_POS) as *mut i32, self.pos as i32);
                std::ptr::write_volatile((p + HIO_DONE) as *mut u32, 0);
                std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            }
            return;
        }
        let b = self.bufs(tokens.len());
        self.htod(b.x, &e);
        self.htod(b.pos, &[self.pos as i32]);
    }

    /// 生成 (1トークンの CUDA Graph) では、入出力を主記憶の hio で受け渡す
    fn host_io(&self, t: usize) -> bool {
        t == 1 && !model::tracing()
    }

    /// (host_io のとき) グラフが終わるのを待つ
    fn wait_done(&self) {
        let flag = (self.hio.1 + HIO_DONE) as *const u32;
        let t0 = Instant::now();
        // SAFETY: 旗は GPU が書くので volatile で読む
        while unsafe { std::ptr::read_volatile(flag) } == 0 {
            std::hint::spin_loop();
            assert!(t0.elapsed().as_secs() < 30, "GPU がトークンの計算を終えない");
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    }

    /// 位置 self.pos から T トークンを処理する。T <= SPEC_T なら、最初の1回でカーネル列を CUDA Graph に記録し、
    /// 以降は再生する。それより多いときとトレース中は、毎回カーネルを投げる。
    /// verify なら全行の最大の logits の位置を b.amax に、そうでなければ最後の行の logits を b.logits に書く
    fn run(&self, t: usize, verify: bool) {
        let key = (t, verify);
        let small = t <= SPEC_T && !model::tracing();
        if small && let Some(gs) = self.graphs.borrow().get(&key) {
            if t == 1 {
                gs[0].launch().unwrap();
                self.serve_host();
                return;
            }
            // OCTO_PROF を付けると、区切りごとに GPU と CPU の時間 (ms) を交互に出す
            let prof = std::env::var_os("OCTO_PROF").is_some();
            let mut ms = vec![];
            for (i, g) in gs.iter().enumerate() {
                let t0 = Instant::now();
                g.launch().unwrap();
                if prof {
                    self.stream.synchronize().unwrap();
                    ms.push(t0.elapsed().as_secs_f64() * 1e3);
                }
                if i + 1 < gs.len() {
                    let t0 = Instant::now();
                    self.cpu_moe(MXFP8_FROM + i, t);
                    ms.push(t0.elapsed().as_secs_f64() * 1e3);
                }
            }
            if prof {
                eprintln!("prof {ms:.2?}");
            }
            return;
        }
        self.stamps.borrow_mut().clear();
        if t > SPEC_T {
            self.take_big();
        }
        self.capturing.set(small);
        if small && t == 1 {
            self.host_seq.borrow_mut().clear();
        }
        self.begin();
        self.record(t, verify);
        self.end();
        self.capturing.set(false);
        if small && t == 1 {
            self.serve_host();
        }
        if small {
            let gs = std::mem::take(&mut *self.recording.borrow_mut());
            self.graphs.borrow_mut().insert(key, gs);
        }
    }

    /// 1トークンを処理して、語彙の logits を返す。
    pub fn forward(&mut self, token: u32) -> Vec<f32> {
        self.inputs(&[token]);
        self.run(1, false);
        self.pos += 1;
        if self.host_io(1) {
            self.wait_done();
            // SAFETY: softcap が主記憶にも書いた logits。旗が立ったので書き終わっている
            return unsafe { std::slice::from_raw_parts((self.hio.1 + HIO_LOGITS) as *const f32, VOCAB) }.to_vec();
        }
        self.dtoh(self.small.logits, VOCAB)
    }

    /// 1トークンを処理して、logits の最大の位置 (貪欲法の次のトークン) を返す。logits は GPU で調べ、位置だけを読む
    pub fn forward_greedy(&mut self, token: u32) -> u32 {
        self.inputs(&[token]);
        self.run(1, false);
        if stamping() {
            // (名前, 合計の µs, 回数)。区切りの先頭 (begin) は CPU の時間を含むので数えない
            let names = self.stamps.borrow();
            let t: Vec<u64> = self.dtoh(self.stamp_buf, names.len());
            let mut agg: Vec<(&str, f64, usize)> = vec![];
            for i in (1..names.len()).filter(|&i| names[i] != "begin") {
                let us = (t[i] - t[i - 1]) as f64 / 1e3;
                match agg.iter_mut().find(|a| a.0 == names[i]) {
                    Some(a) => (a.1, a.2) = (a.1 + us, a.2 + 1),
                    None => agg.push((names[i], us, 1)),
                }
            }
            agg.sort_by(|a, b| b.1.total_cmp(&a.1));
            eprintln!("stamp {}", agg.iter().map(|a| format!("{}:{:.0}us/{}", a.0, a.1, a.2)).collect::<Vec<_>>().join(" "));
        }
        self.pos += 1;
        let best = if self.host_io(1) {
            self.wait_done();
            // SAFETY: done_host が書いた値
            unsafe { std::ptr::read_volatile((self.hio.1 + HIO_BEST) as *const u64) }
        } else {
            self.dtoh::<u64>(self.small.best, 1)[0]
        };
        0xffff_ffff - best as u32
    }

    /// プロンプトを TMAX トークンずつまとめて処理し、最後のトークンの logits を返す。
    /// after(chunk の長さ) はチャンクごとに呼ばれ、その間 taps にそのチャンクの層出力がある
    pub fn prefill_with(&mut self, tokens: &[u32], mut after: impl FnMut(&Self, usize)) -> Vec<f32> {
        for chunk in tokens.chunks(TMAX) {
            self.inputs(chunk);
            self.run(chunk.len(), false);
            after(self, chunk.len());
            self.pos += chunk.len();
        }
        self.dtoh(self.bufs(tokens.len().saturating_sub(1) % TMAX + 1).logits, VOCAB)
    }

    pub fn prefill(&mut self, tokens: &[u32]) -> Vec<f32> {
        self.prefill_with(tokens, |_, _| {})
    }

    /// (投機的デコード) tokens を位置 self.pos から一度に処理し、各行の次のトークン (logits の最大の位置) を返す。
    /// 外れた下書きは rewind で捨てる
    pub fn verify(&mut self, tokens: &[u32]) -> Vec<u32> {
        assert!(tokens.len() <= SPEC_T);
        self.inputs(tokens);
        self.run(tokens.len(), true);
        self.pos += tokens.len();
        self.dtoh::<i32>(self.small.amax, tokens.len()).into_iter().map(|t| t as u32).collect()
    }

    /// 直前の verify の t トークンのうち、先頭 keep 個だけを残す。畳み込みの状態を戻し、位置を下げる。
    /// KV キャッシュは位置で読むので、捨てた位置は次に書くときに上書きされる
    pub fn rewind(&mut self, keep: usize, t: usize) {
        if keep < t {
            k!(self, "conv_rewind", (D.div_ceil(256), self.slots.len() * 4), 256, 0; self.rewind_tab, keep as i32);
            self.pos -= t - keep;
        }
    }

    /// T トークンを処理するときの作業領域
    fn bufs(&self, t: usize) -> &Bufs {
        if t <= SPEC_T { &self.small } else { &self.big }
    }

    /// (下書きモデル) 直前に処理した n トークンの、下書きモデルが読む層出力の場所と、種類ごとのずれ (行数)
    pub(crate) fn taps(&self, n: usize) -> (P, usize) {
        let b = self.bufs(n);
        (b.taps, b.cap)
    }

    /// 次に処理するトークンの位置 (= ここまでに処理したトークン数)
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// 文脈長の上限
    pub fn max_ctx(&self) -> usize {
        self.ctx
    }

    /// 文脈を捨てて位置 0 から始め直す。畳み込みが持つ直前3トークンの入力を 0 に戻す。
    /// KV キャッシュは、注意が今の文脈の位置しか読まないので消さなくてよい
    pub fn reset(&mut self) {
        for s in &self.slots {
            for (&h, ch) in s.hist.iter().zip([1024, 1024, D, D]) {
                // SAFETY: hist は ch * 3 個の f32 で確保してある
                unsafe { result::memset_d8_async(h, 0, ch * 3 * 4, self.stream.cu_stream()) }.unwrap();
            }
        }
        self.stream.synchronize().unwrap();
        self.pos = 0;
    }

    fn begin(&self) {
        if self.capturing.get() {
            self.stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL).unwrap();
            if stamping() {
                self.stamp("begin");
            }
        }
    }

    fn end(&self) {
        if self.capturing.get() {
            let g = self.stream.end_capture(sys::CUgraphInstantiate_flags(0)).unwrap().expect("空のグラフ");
            g.launch().unwrap();
            self.recording.borrow_mut().push(g);
        }
    }

    /// (生成と検証) 20〜23層の MoE を CPU で計算する。b.h を読み b.y に書く
    fn cpu_moe(&self, il: usize, t: usize) {
        let h: Vec<f32> = self.dtoh(self.small.h, t * D);
        let l = &self.cpu.layers[il];
        if t == 1 {
            // (トレース中だけ) グラフを使わないので、全部 CPU で計算する
            return self.htod(self.small.y, &self.cpu.moe(l, &h));
        }
        // ここで投げるカーネルは CUDA Graph に入らない
        let capturing = self.capturing.replace(false);
        let routes: Vec<_> = h.chunks(D).map(|x| self.cpu.route(l, x)).collect();
        let ids: Vec<i32> = routes.iter().flat_map(|r| r.0.map(|e| e as i32)).collect();
        let wts: Vec<f32> = routes.iter().flat_map(|r| r.1).collect();
        let (b, li) = (&self.small, il - MXFP8_FROM);
        let mut xc = self.xc.borrow_mut();
        self.xc_tick(&mut xc, li);
        for &e in &ids {
            xc.cnt[li][e as usize] += 1.0;
        }
        // 枠にある expert は GPU、残りは CPU。expert ごとに割り当てをまとめ、重みの読み出しを共有する
        let (mut on_gpu, mut on_cpu) = ([false; N_EXP], vec![]);
        let (mut order, mut off, mut u) = (vec![], vec![-1i32; 129], 0);
        for ex in 0..N_EXP {
            let asg: Vec<i32> = (0..ids.len()).filter(|&a| ids[a] as usize == ex).map(|a| a as i32).collect();
            if asg.is_empty() {
                continue;
            }
            match xc.slot[li][ex] {
                Some(s) if xc.ready[s] => {
                    on_gpu[ex] = true;
                    off[u] = order.len() as i32;
                    off[65 + u] = s as i32;
                    order.extend(asg);
                    u += 1;
                }
                _ => on_cpu.push(ex),
            }
        }
        off[u] = order.len() as i32;
        if u > 0 {
            self.htod(b.order, &order);
            self.htod(b.off, &off);
            self.htod(b.wts, &wts);
            // SAFETY: yd は T*8*D 要素ある。CPU が受け持つ割り当ての行は 0 のまま moe_sum に渡る
            unsafe { result::memset_d8_async(b.yd, 0, t * TOP_K * D * 4, self.stream.cu_stream()) }.unwrap();
            k!(self, "moe_up_mx_g", (u * FF * 32).div_ceil(256), 256, 0; xc.base, b.order, b.off, b.h, b.act);
            k!(self, "moe_down_mx_g", (u * D * 32).div_ceil(256), 256, 0; xc.base, b.order, b.off, b.act, b.yd);
            k!(self, "moe_sum", (t * D).div_ceil(256), 256, 0; b.yd, b.wts, b.y, t as i32);
        }
        // GPU が計算しているあいだに、残りを CPU で計算する
        let y = self.cpu.moe_batch(l, &h, &ids, &wts, |ex| !on_gpu[ex]);
        if u > 0 {
            self.htod(b.a, &y);
            k!(self, "add", (t * D).div_ceil(256), 256, 0; b.y, b.a, b.y, (t * D) as i32);
        } else {
            self.htod(b.y, &y);
        }
        if li == N_LAYER - MXFP8_FROM - 1 {
            self.token_done.record(&self.stream).unwrap();
        }
        self.plan_fills(&mut xc, li, on_cpu);
        self.capturing.set(capturing);
    }

    /// 1トークン (1回の検証) の最初の層で、選ばれた回数を減らし、届いた枠を使えるようにする
    fn xc_tick(&self, xc: &mut XCache, li: usize) {
        if li == 0 {
            xc.cnt.iter_mut().flatten().for_each(|c| *c *= XDECAY);
            if !xc.issued.is_empty() && self.fill_done.is_complete() {
                for s in std::mem::take(&mut xc.issued) {
                    xc.ready[s] = true;
                }
            }
        }
    }

    /// CPU で計算した expert のうち選ばれた回数の多いものを、空いた枠か回数の最も少ない枠へ送ると決める。
    /// 転送は、GPU がまだ枠を読んでいるかもしれないので、token_done のあとにまとめて出す
    fn plan_fills(&self, xc: &mut XCache, li: usize, mut on_cpu: Vec<usize>) {
        on_cpu.sort_by(|&a, &c| xc.cnt[li][c].total_cmp(&xc.cnt[li][a]));
        for &ex in on_cpu.iter().take(XFILLS) {
            let score = |xc: &XCache, s: usize| xc.owner[s].map_or(-1.0, |(l2, e2)| xc.cnt[l2][e2]);
            let victim = (0..xc.owner.len()).min_by(|&a, &c| score(xc, a).total_cmp(&score(xc, c))).unwrap();
            if score(xc, victim) >= xc.cnt[li][ex] {
                break;
            }
            if let Some((l2, e2)) = xc.owner[victim] {
                xc.slot[l2][e2] = None;
            }
            xc.owner[victim] = Some((li, ex));
            xc.slot[li][ex] = Some(victim);
            xc.ready[victim] = false;
            xc.pending.push((li, ex, victim));
        }
        if li == N_LAYER - MXFP8_FROM - 1 && !xc.pending.is_empty() {
            self.copy.wait(&self.token_done).unwrap();
            for (li, ex, victim) in std::mem::take(&mut xc.pending) {
                let e = &self.cpu.layers[MXFP8_FROM + li].experts;
                let mut at = xc.base + (victim * XSLOT) as u64;
                for (src, size) in [(e.w13, W13), (e.s13, S13), (e.w2, W2), (e.s2, S2)] {
                    // SAFETY: src はページ固定メモリ。枠は XSLOT バイトある
                    unsafe { result::memcpy_htod_async(at, &src[ex * size..(ex + 1) * size], self.copy.cu_stream()) }.unwrap();
                    at += size as u64;
                }
                xc.issued.push(victim);
            }
            self.fill_done.record(&self.copy).unwrap();
        }
    }

    /// (生成) 1トークン分の CUDA Graph を起動したあとに呼ぶ。グラフは 20〜23層の MoE の入力 h を主記憶に書いて待っているので、
    /// 層ごとに、選ばれた expert のうち枠にあるものを route で GPU に頼み、残りを CPU で計算して y に書く。
    /// OCTO_PROF を付けると、層ごとに h を待った時間と CPU の時間 (ms)、最後に残りの GPU の時間を出す
    fn serve_host(&self) {
        self.token_done.record(&self.stream).unwrap();
        let prof = std::env::var_os("OCTO_PROF").is_some();
        let mut ms = vec![];
        let seq = self.host_seq.borrow().clone();
        for ev in seq {
            let li = match ev {
                HostEv::Attn(slot) => {
                    self.host_attn(slot);
                    continue;
                }
                HostEv::Moe(li) => li,
            };
            let t0 = Instant::now();
            let h = self.host_h(li);
            let t1 = Instant::now();
            self.host_layer(li, &h);
            if prof {
                ms.extend([(t1 - t0).as_secs_f64() * 1e3, t1.elapsed().as_secs_f64() * 1e3]);
            }
        }
        if prof {
            let t0 = Instant::now();
            self.stream.synchronize().unwrap();
            ms.push(t0.elapsed().as_secs_f64() * 1e3);
            eprintln!("prof {ms:.2?}");
        }
    }

    /// (生成、主記憶に KV がある global 層) GPU が書いた q, r, sink を待ち、KV 全体への注意を CPU で計算して返す
    fn host_attn(&self, slot: usize) {
        let at = self.ha.1 + slot * HA;
        let flag = (at + HA_FQ) as *mut u32;
        let [kh, vh] = self.slots[slot].host.unwrap();
        let rel = model::bf_vec(self.cpu.layers[phys(slot)].rel_proj);
        let t0 = Instant::now();
        // SAFETY: ha は主記憶の受け渡しの場所。GPU は旗が立つまで結果を読まない。KV は pos + 1 行まで書かれている
        unsafe {
            while std::ptr::read_volatile(flag) == 0 {
                std::hint::spin_loop();
                assert!(t0.elapsed().as_secs() < 30, "GPU が注意の入力を書かない");
            }
            std::ptr::write_volatile(flag, 0);
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            let qr = std::slice::from_raw_parts((at + HA_Q) as *const f32, 3480);
            let out = std::slice::from_raw_parts_mut((at + HA_O) as *mut f32, 3072);
            host_attn::attend(&qr[..3072], &qr[3072..3456], &qr[3456..], kh as *const u16, vh as *const u16, self.pos + 1, &rel, out);
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            std::ptr::write_volatile((at + HA_FO) as *mut u32, 1);
        }
    }

    /// 主記憶の、層 li の受け渡しの場所の off バイト目
    fn hx_at<T>(&self, li: usize, off: usize) -> *mut T {
        (self.hx.1 + li * HX + off) as *mut T
    }

    /// GPU が層 li の h を書くのを待って読む
    fn host_h(&self, li: usize) -> Vec<f32> {
        let flag = self.hx_at::<u32>(li, HX_FH);
        let t0 = Instant::now();
        // SAFETY: hx は主記憶に確保した受け渡しの場所。旗は GPU が書くので volatile で読む
        unsafe {
            while std::ptr::read_volatile(flag) == 0 {
                std::hint::spin_loop();
                assert!(t0.elapsed().as_secs() < 30, "GPU が 20〜23層の入力を書かない");
            }
            std::ptr::write_volatile(flag, 0);
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            std::slice::from_raw_parts(self.hx_at::<f32>(li, HX_H), D).to_vec()
        }
    }

    /// 層 li (20 + li 層) の MoE。選ばれた expert のうち枠にあって中身が届いているものを GPU に頼み (route)、残りを同時に CPU で計算して y に書く。
    /// そのあと、CPU で計算した expert のうち選ばれた回数の多いものを、空いた枠か回数の最も少ない枠へ送ると決める。
    /// 転送は、CPU の計算と主記憶の帯域を取り合わず、GPU がまだ枠を読んでいるかもしれないので、このトークンのグラフが終わってから出す
    fn host_layer(&self, li: usize, h: &[f32]) {
        let l = &self.cpu.layers[MXFP8_FROM + li];
        let (sel, wts) = self.cpu.route(l, h);
        let mut xc = self.xc.borrow_mut();
        self.xc_tick(&mut xc, li);
        for &e in &sel {
            xc.cnt[li][e] += 1.0;
        }
        let (mut slots, mut g, mut n) = (0u64, [0.0f32; TOP_K], 0);
        let (mut on_cpu, mut cpu_wts) = (vec![], vec![]);
        for (&e, &w) in sel.iter().zip(&wts) {
            match xc.slot[li][e] {
                Some(s) if xc.ready[s] => {
                    slots |= (s as u64) << (8 * n);
                    g[n] = w;
                    n += 1;
                }
                _ => {
                    on_cpu.push(e);
                    cpu_wts.push(w);
                }
            }
        }
        // SAFETY: hx は主記憶の受け渡しの場所。GPU は旗が立つまで route と y を読まない
        unsafe {
            std::ptr::write_volatile(self.hx_at::<u32>(li, HX_ROUTE), n as u32);
            std::ptr::write_volatile(self.hx_at::<u64>(li, HX_ROUTE + 8), slots);
            std::ptr::copy_nonoverlapping(g.as_ptr(), self.hx_at::<f32>(li, HX_ROUTE + 16), TOP_K);
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            std::ptr::write_volatile(self.hx_at::<u32>(li, HX_FR), 1);
        }
        let y = if on_cpu.is_empty() { vec![0.0; D] } else { self.cpu.moe_some(l, h, &on_cpu, &cpu_wts) };
        // SAFETY: 同上
        unsafe {
            std::ptr::copy_nonoverlapping(y.as_ptr(), self.hx_at::<f32>(li, HX_Y), D);
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            std::ptr::write_volatile(self.hx_at::<u32>(li, HX_FY), 1);
        }
        self.plan_fills(&mut xc, li, on_cpu);
    }

    /// prefill で大きい作業領域と置き場 (expert の先頭の reserved 枠) を使う前に呼ぶ。枠への転送が終わるのを待ち、その枠を空にする
    fn take_big(&self) {
        self.stream.wait(&self.fill_done).unwrap();
        let mut xc = self.xc.borrow_mut();
        // 待ったので、出した転送は全部届く
        for s in std::mem::take(&mut xc.issued) {
            xc.ready[s] = true;
        }
        for s in 0..xc.reserved {
            if let Some((l2, e2)) = xc.owner[s].take() {
                xc.slot[l2][e2] = None;
            }
            xc.ready[s] = false;
        }
    }


    /// (prefill) 20〜23層の MoE を、expert 8個ずつの組に分け、一部の組は PCIe で送って GPU で、残りは CPU で同時に計算する。
    /// b.h を読み b.y に書く
    fn hybrid_moe(&self, il: usize, t: usize) {
        let (b, l, cl) = (&self.big, &self.layers[il], &self.cpu.layers[il]);
        let n = t * TOP_K;
        self.gemm(l.router, b.h, b.rl, 128, D, t);
        k!(self, "topk", t, 128, 0; b.rl, l.router_bias, b.ids, b.wts);
        k!(self, "moe_sort", 1, 32, 0; b.ids, n as i32, b.order, b.off);
        // SAFETY: yd は T*8*D 要素ある。CPU が受け持つ割り当ての行は 0 のまま moe_sum に渡る
        unsafe { result::memset_d8_async(b.yd, 0, n * D * 4, self.stream.cu_stream()) }.unwrap();
        let ids: Vec<i32> = self.dtoh(b.ids, n);
        let wts: Vec<f32> = self.dtoh(b.wts, n);
        let off: Vec<i32> = self.dtoh(b.off, N_EXP + 1);
        let h: Vec<f32> = self.dtoh(b.h, t * D);

        // GPU の手間は組の数、CPU の手間は割り当ての数に比例するとみて、割り当ての多い組から GPU に回す
        let count = |g: usize| (off[(g + 1) * GROUP] - off[g * GROUP]) as f64;
        let mut groups: Vec<usize> = (0..N_EXP / GROUP).filter(|&g| count(g) > 0.0).collect();
        groups.sort_by(|&a, &b| count(b).total_cmp(&count(a)));
        let mut cpu_cost = groups.iter().map(|&g| count(g)).sum::<f64>() * CPU_PER_ASSIGN;
        let mut k = 0;
        while k < groups.len() {
            let next = cpu_cost - count(groups[k]) * CPU_PER_ASSIGN;
            if ((k + 1) as f64 * GPU_PER_GROUP).max(next) >= (k as f64 * GPU_PER_GROUP).max(cpu_cost) {
                break;
            }
            cpu_cost = next;
            k += 1;
        }
        let mut on_gpu = [false; N_EXP];
        for &g in &groups[..k] {
            on_gpu[g * GROUP..(g + 1) * GROUP].fill(true);
        }

        let e = &cl.experts;
        let at = [0, GROUP * W13, GROUP * (W13 + S13), GROUP * (W13 + S13 + W2)];
        for (i, &g) in groups[..k].iter().enumerate() {
            let (s, e0) = (i % 2, g * GROUP);
            let st = self.staging[s];
            // 置き場 s を前に使った計算が終わってから上書きする
            self.copy.wait(&self.comp_done[s]).unwrap();
            for ((src, size), at) in [(e.w13, W13), (e.s13, S13), (e.w2, W2), (e.s2, S2)].into_iter().zip(at) {
                // SAFETY: src はページ固定メモリなので非同期に送れる。置き場は GROUP 個分ある
                unsafe { result::memcpy_htod_async(st + at as u64, &src[e0 * size..(e0 + GROUP) * size], self.copy.cu_stream()) }.unwrap();
            }
            self.up_done[s].record(&self.copy).unwrap();
            self.stream.wait(&self.up_done[s]).unwrap();
            let [w13, s13, w2, s2] = at.map(|a| st + a as u64);
            let (base, cnt) = (off[e0], off[e0 + GROUP] - off[e0]);
            k!(self, "gemm_mxfp8_grouped", (2 * FF / 64, GROUP), 128, 0;
                w13, s13, (2 * FF) as i32, D as i32, b.h, TOP_K as i32, b.order, b.off, e0 as i32, b.gu);
            k!(self, "situ_sel", (cnt as usize * FF).div_ceil(256), 256, 0; b.gu, b.act, b.order, base, cnt);
            k!(self, "gemm_mxfp8_grouped", (D / 64, GROUP), 128, 0; w2, s2, D as i32, FF as i32, b.act, 1i32, b.order, b.off, e0 as i32, b.yd);
            self.comp_done[s].record(&self.stream).unwrap();
        }
        // GPU が送って計算しているあいだに、残りの組を CPU で計算する
        let y = self.cpu.moe_batch(cl, &h, &ids, &wts, |ex| !on_gpu[ex]);
        self.htod(b.a, &y);
        k!(self, "moe_sum", (t * D).div_ceil(256), 256, 0; b.yd, b.wts, b.y, t as i32);
        k!(self, "add", (t * D).div_ceil(256), 256, 0; b.y, b.a, b.y, (t * D) as i32);
    }

    /// T トークン分のカーネル列を投げる。logits は、verify なら全行の最大の位置だけ、そうでなければ最後の行を求める
    fn record(&self, t: usize, verify: bool) {
        let b = self.bufs(t);
        let io = t == 1 && self.capturing.get();
        if io {
            k!(self, "from_host", D.div_ceil(256), 256, 0; self.hio.0 + HIO_X as u64, self.hio.0 + HIO_POS as u64, b.x, b.pos);
        }
        let slab = |i: usize| f4(i * t * D); // [種類][T][D] の i 番目
        let mut nb = 0;
        for il in 0..CORE.0 {
            self.layer(il, il, il, &mut nb, None, t);
        }
        self.copy(b.prelude, b.x, t * D);
        let prelude_nb = nb;
        for pass in 0..PASSES {
            if pass > 0 {
                // 前の周の、最初のコアブロックより後のブロックと出力
                assert_eq!(nb, prelude_nb + 2);
                self.copy(b.bank, b.blocks + slab(prelude_nb + 1), t * D);
                self.copy(b.bank + slab(1), b.x, t * D);
                self.copy(b.x, b.prelude, t * D);
                nb = prelude_nb;
            }
            for ip in CORE.0..CORE.1 {
                let slot = if pass == 0 { ip } else { N_LAYER + (pass - 1) * (CORE.1 - CORE.0) + ip - CORE.0 };
                // メッセージはその周のバンクだけから決まる。小さいバッチでは 12 層分を周の始めに1回で作る
                let msgs = if pass == 0 {
                    None
                } else if t <= SPEC_T {
                    if ip == CORE.0 {
                        self.mix(b.bank, 1, b.bank + slab(1), self.msg_queries, 4 * (CORE.1 - CORE.0), b.msgs, t, None, if t == 1 { b.rms + f4(4) } else { 0 });
                    }
                    Some(b.msgs + slab(4 * (ip - CORE.0)))
                } else {
                    self.mix(b.bank, 1, b.bank + slab(1), self.layers[ip].msg_query, 4, b.msgs, t, None, 0);
                    Some(b.msgs)
                };
                self.layer(ip, ip, slot, &mut nb, msgs, t);
            }
        }
        for il in CORE.1..N_LAYER {
            self.layer(il, il, il, &mut nb, None, t);
        }
        self.mix(b.blocks, nb, b.x, self.output_query, 1, b.h, t, None, 0);
        // 正規化した最後の隠れ状態は、下書きモデルの4つ目の入力でもある
        let normed = b.taps + f4(3 * b.cap * D);
        k!(self, "rmsnorm", t, 256, 0; b.h, self.norm, normed, 0i32, t as i32);
        if verify {
            // softcap は単調なので、最大の位置は変わらない
            self.gemm(self.unembed, normed, b.logits_all, 201024, D, t);
            k!(self, "argmax_rows", t, 1024, 0; b.logits_all, 201024i32, VOCAB as i32, b.amax);
        } else {
            self.gemm(self.unembed, normed + f4((t - 1) * D), b.logits, 201024, D, 1);
            // SAFETY: best は 8 バイトある
            unsafe { result::memset_d8_async(b.best, 0, 8, self.stream.cu_stream()) }.unwrap();
            let host = if io { self.hio.0 + HIO_LOGITS as u64 } else { 0 };
            k!(self, "softcap", VOCAB.div_ceil(256), 256, 0; b.logits, VOCAB as i32, b.best, host);
            if io {
                k!(self, "done_host", 1, 32, 0; b.best, self.hio.0 + HIO_BEST as u64, self.hio.0 + HIO_DONE as u64);
            }
        }
    }

    /// (prefill、主記憶の KV) 位置 0..pos+t の KV を KB 位置ずつ置き場に送り、塊ごとに注意を足し込む。
    /// GPU が主記憶を直接読むと、クエリのタイルごとに同じ KV を読み直すことになり遅い
    fn attn_host(&self, host: [usize; 2], l: &GLayer, t: usize) {
        // 置き場 1 つ (65.7 MB) に K と V が入る位置数 (1 位置 2 KB ずつ)。16 の倍数に切り下げる
        const KB: usize = GROUP * (W13 + S13 + W2 + S2) / 4096 / 16 * 16;
        let (b, st, end) = (&self.big, self.staging[0], self.pos + t);
        for (i, b0) in (0..end).step_by(KB).enumerate() {
            let nb = KB.min(end - b0);
            for (j, &h) in host.iter().enumerate() {
                // SAFETY: host はページ固定の主記憶で、end 行以上ある。置き場は K と V で 2 * KB 行ある
                unsafe {
                    let src = std::slice::from_raw_parts((h as *const u16).add(b0 * 1024), nb * 1024);
                    result::memcpy_htod_async(st + (j * KB * 2048) as u64, src, self.stream.cu_stream()).unwrap();
                }
            }
            k!(self, "attn_blk", (8, t.div_ceil(32)), 128, 0;
                b.qx, st, st + (KB * 2048) as u64, b0 as i32, nb as i32, l.rel_proj, b.acc, b.pos, t as i32, (i == 0) as i32);
        }
        k!(self, "attn_fin", (24, t), 128, 0; b.qx, b.acc, b.o);
    }

    #[allow(clippy::too_many_arguments)]
    /// msgs はループの 2 周目以降のメッセージ [4][T][D] の場所
    fn layer(&self, il: usize, ip: usize, slot: usize, nb: &mut usize, msg: Option<P>, t: usize) {
        let msgs = msg.is_some();
        let (l, s, b) = (&self.layers[il], &self.slots[slot], self.bufs(t));
        let slab = |i: usize| f4(i * t * D);
        let ti = t as i32;
        // attention の入力: SWA は1本、global は Q, K, V に1本ずつ
        // control がなければ、正規化も mix のカーネルでかける
        let wstride = if l.global { D } else { 0 };
        let fused = il > 0 && !msgs;
        if il > 0 {
            self.mix(b.blocks, *nb, b.x, l.attn_query, if l.global { 3 } else { 1 }, b.inp, t, fused.then_some((l.norms, wstride)), if msgs && t == 1 { b.rms } else { 0 });
        } else {
            self.copy(b.inp, b.x, t * D);
        }
        let separate = l.global || msgs;
        if msgs {
            // SWA は1本の入力から Q, K, V の3本を作る
            let mr = b.rms + f4(4 + 4 * (ip - CORE.0));
            self.control(b.inp, !l.global, b.inp, msg.unwrap(), &l.ctrl[..3], t, b.rms, mr);
        }
        let n_in = if separate { 3 } else { 1 };
        if !fused {
            k!(self, "rmsnorm", n_in * t, 256, 0; b.inp, l.norms, b.inp, wstride as i32, ti);
        }
        let (xq, xk, xv) = if separate { (b.inp, b.inp + slab(1), b.inp + slab(2)) } else { (b.inp, b.inp, b.inp) };

        let nk = if l.global { 1024 + 32 } else { 1024 };
        if t == 1 {
            k!(self, "mv3_bf16", ((QX + 2 * nk) * 32).div_ceil(256), 256, 0;
                l.wq, xq, b.qx, QX as i32, l.wk, xk, b.kx, nk as i32, l.wv, xv, b.vx, nk as i32);
        } else {
            self.gemm(l.wq, xq, b.qx, QX, D, t);
            self.gemm(l.wk, xk, b.kx, nk, D, t);
            self.gemm(l.wv, xv, b.vx, nk, D, t);
        }
        let dyn_ = l.global.then_some(1024);
        // 小さいバッチ (生成と検証) では、外れた下書きを捨てられるよう畳み込みの入力を残す
        let seq = |k: usize| if t <= SPEC_T { s.seq[k] } else { 0 };
        let ring = if l.global { 0 } else { RING as i32 };
        if t == 1 {
            k!(self, "qkv_conv_post", 64, 128, 0; b.qx, b.kx, b.vx, l.k_conv, l.v_conv, dyn_.map_or(-1, |d| d as i32),
                s.hist[0], s.hist[1], s.seq[0], s.seq[1], l.q_clip, l.rel_clip, b.pos, ring, s.kc, s.vc, s.host.is_some() as i32);
        } else {
            self.conv(b.kx, nk, l.k_conv, dyn_, s.hist[0], b.kp, 1024, 1024, 128, t, seq(0), 0);
            self.conv(b.vx, nk, l.v_conv, dyn_, s.hist[1], b.vp, 1024, 1024, 128, t, seq(1), 0);
            k!(self, "qkv_post", (64, t), 128, 0; b.qx, b.kp, b.vp, l.q_clip, l.rel_clip, b.pos, ring, s.kc, s.vc);
        }
        if t == 1 && s.host.is_some() && self.capturing.get() {
            // 主記憶の KV は CPU が読んで注意を計算する (serve_host)
            let at = |off: usize| self.ha.0 + (slot * HA + off) as u64;
            self.host_seq.borrow_mut().push(HostEv::Attn(slot));
            k!(self, "attn_to_host", 1, 256, 0; b.qx, at(HA_Q), at(HA_FQ));
            k!(self, "wait_host", 1, 32, 0; at(HA_FO), 0u64, 0u64, 0i32);
            k!(self, "attn_from_host", 12, 256, 0; b.qx, at(HA_O), b.o);
        } else if t <= SPEC_T {
            // 検証のバッチは、主記憶の KV も GPU から直接読む
            k!(self, "attn_part", (8, 32 * t), 128, 0; b.qx, s.kc, s.vc, l.rel_proj, b.part, b.pos, ring, b.o, b.done);
        } else if let Some(host) = s.host {
            self.attn_host(host, l, t);
        } else {
            k!(self, "attn_tile", (8, t.div_ceil(32)), 128, 0; b.qx, s.kc, s.vc, l.rel_proj, b.o, b.pos, ring, ti);
        }
        self.gemm(l.wo, b.o, b.a, D, 3072, t);
        let start = ip.is_multiple_of(BLOCK);
        self.conv(b.a, D, l.attn_conv, None, s.hist[2], b.a, D, D, D, t, seq(2), if start { 0 } else { b.x });
        // ブロックの先頭層は残差を引き継がず、入力をブロック列に積む
        if start {
            self.copy(b.blocks + slab(*nb), b.x, t * D);
            *nb += 1;
            self.copy(b.x, b.a, t * D);
        }

        let mlp_fused = !msgs;
        self.mix(b.blocks, *nb, b.x, l.mlp_query, 1, b.h, t, mlp_fused.then_some((l.mlp_norm, 0)), if msgs && t == 1 { b.rms + f4(3) } else { 0 });
        if msgs {
            self.control(b.h, false, b.h, msg.unwrap() + slab(3), &l.ctrl[3..], t, b.rms + f4(3), b.rms + f4(4 + 4 * (ip - CORE.0) + 3));
        }
        if !mlp_fused {
            k!(self, "rmsnorm", t, 256, 0; b.h, l.mlp_norm, b.h, 0i32, ti);
        }
        if !matches!(l.experts, GExperts::Host) {
            if t == 1 {
                k!(self, "router_topk", 64, 256, 0; l.router, b.h, b.rl, l.router_bias, b.ids, b.wts, b.done + f4(SPEC_T * 24));
            } else {
                self.gemm(l.router, b.h, b.rl, 128, D, t);
                k!(self, "topk", t, 128, 0; b.rl, l.router_bias, b.ids, b.wts);
            }
        }
        match l.experts {
            GExperts::Sk { tab, ixd } => {
                if t <= SPEC_T {
                    k!(self, "moe_up_sk", (t * TOP_K * FF / 4 * 32).div_ceil(256), 256, 0; tab, ixd, b.ids, b.h, b.act, ti);
                    k!(self, "moe_down_sk", (t * D / 2 * 32).div_ceil(256), 256, 0; tab, b.ids, b.wts, b.act, b.y, ti);
                } else {
                    let n = (t * TOP_K) as i32;
                    k!(self, "moe_sort", 1, 32, 0; b.ids, n, b.order, b.off);
                    k!(self, "gemm_sk_grouped", (2 * FF / 64, 128), 128, 0; tab, 0i32, (2 * FF) as i32, D as i32, b.h, TOP_K as i32, b.order, b.off, b.gu);
                    k!(self, "situ_sk", (t * TOP_K * FF).div_ceil(256), 256, 0; b.gu, b.act, b.ids, ixd, n);
                    k!(self, "gemm_sk_grouped", (D / 64, 128), 128, 0; tab, 1i32, D as i32, FF as i32, b.act, 1i32, b.order, b.off, b.yd);
                    k!(self, "moe_sum", (t * D).div_ceil(256), 256, 0; b.yd, b.wts, b.y, ti);
                }
            }
            GExperts::Fp4([w13, s13, a13, w2, s2, a2]) => {
                if t <= SPEC_T {
                    k!(self, "moe_up_nvfp4", (t * TOP_K * FF / 4 * 32).div_ceil(256), 256, 0; w13, s13, a13, b.ids, b.h, b.act, ti);
                    k!(self, "moe_down_nvfp4", (t * D / 2 * 32).div_ceil(256), 256, 0; w2, s2, a2, b.ids, b.wts, b.act, b.y, ti);
                } else {
                    let n = (t * TOP_K) as i32;
                    k!(self, "moe_sort", 1, 32, 0; b.ids, n, b.order, b.off);
                    k!(self, "gemm_nvfp4_grouped", (2 * FF / 64, 128), 128, 0;
                        w13, s13, a13, (2 * FF) as i32, D as i32, b.h, TOP_K as i32, b.order, b.off, b.gu);
                    k!(self, "situ_rows", (t * TOP_K * FF).div_ceil(256), 256, 0; b.gu, b.act, n);
                    k!(self, "gemm_nvfp4_grouped", (D / 64, 128), 128, 0; w2, s2, a2, D as i32, FF as i32, b.act, 1i32, b.order, b.off, b.yd);
                    k!(self, "moe_sum", (t * D).div_ceil(256), 256, 0; b.yd, b.wts, b.y, ti);
                }
            }
            GExperts::Host if t == 1 && self.capturing.get() => {
                // CPU との受け渡しをグラフの中で待つ (serve_host)。枠にある expert は GPU で計算し、CPU の分と足す
                let at = |off: usize| self.hx.0 + ((il - MXFP8_FROM) * HX + off) as u64;
                let base = self.xc.borrow().base;
                self.host_seq.borrow_mut().push(HostEv::Moe(il - MXFP8_FROM));
                k!(self, "to_host", 1, 256, 0; b.h, at(HX_H), at(HX_FH));
                k!(self, "wait_host", 1, 32, 0; at(HX_FR), at(HX_ROUTE), b.route, 12i32);
                k!(self, "moe_up_mx", (TOP_K * FF * 32).div_ceil(256), 256, 0; base, b.route, b.h, b.act);
                k!(self, "moe_down_mx", (D * 32).div_ceil(256), 256, 0; base, b.route, b.act, b.yd);
                k!(self, "wait_host", 1, 32, 0; at(HX_FY), 0u64, 0u64, 0i32);
                k!(self, "add_host", D.div_ceil(256), 256, 0; b.yd, at(HX_Y), b.y);
            }
            GExperts::Host if t <= SPEC_T => {
                self.end();
                self.cpu_moe(il, t);
                self.begin();
            }
            GExperts::Host => self.hybrid_moe(il, t),
        }
        self.conv(b.y, D, l.mlp_conv, None, s.hist[3], b.y, D, D, D, t, seq(3), b.x);
        if let Some(j) = TAP_SLOTS.iter().position(|&s| s == slot) {
            self.copy(b.taps + f4(j * b.cap * D), b.x, t * D);
        }
        if model::tracing() && t == 1 {
            model::trace(slot, &self.dtoh(b.x, D));
        }
    }
}

/// OCTO_STAMP を付けると、生成の CUDA Graph の各カーネルのあとに GPU の時刻を記録し、トークンごとにカーネルの種類別の時間 (直前の記録からの差) を出す
fn stamping() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var_os("OCTO_STAMP").is_some());
    *ON
}

fn phys(slot: usize) -> usize {
    if slot < N_LAYER { slot } else { CORE.0 + (slot - N_LAYER) % (CORE.1 - CORE.0) }
}

/// ページ固定のバッファを経由して VRAM に送る。mmap した領域から直接送ると 0.07 GB/s しか出ない。
struct Uploader {
    stream: Arc<CudaStream>,
    pin: *mut u8,
    total: usize,
}

const PIN: usize = 64 << 20;

impl Uploader {
    fn new(stream: &Arc<CudaStream>) -> Self {
        // SAFETY: 確保に失敗したらエラーで止まる。解放はしない (プロセスの終わりまで使う)
        let pin = unsafe { result::malloc_host(PIN, 0) }.expect("ページ固定メモリを確保できない") as *mut u8;
        Self { stream: stream.clone(), pin, total: 0 }
    }

    fn alloc(&mut self, n: usize) -> P {
        // SAFETY: 確保するだけ。解放はしない
        let p = unsafe { result::malloc_sync(n.max(4)) }.unwrap_or_else(|e| panic!("VRAM を {n} バイト確保できない: {e}"));
        self.total += n;
        p
    }

    fn zeros(&mut self, n: usize) -> P {
        let p = self.alloc(n);
        // SAFETY: p は n.max(4) バイト確保してある
        unsafe { result::memset_d8_async(p, 0, n.max(4), self.stream.cu_stream()) }.unwrap();
        p
    }

    fn bytes(&mut self, src: &[u8]) -> P {
        let p = self.alloc(src.len());
        self.to(p, src);
        p
    }

    /// 確保済みの dst に送る。送り終わるまで待つ
    fn to(&mut self, dst: P, src: &[u8]) {
        for (i, c) in src.chunks(PIN).enumerate() {
            // SAFETY: pin は PIN バイトある。前の転送が終わってから上書きする
            unsafe {
                par_copy(c, self.pin);
                let staged = std::slice::from_raw_parts(self.pin, c.len());
                result::memcpy_htod_async(dst + (i * PIN) as u64, staged, self.stream.cu_stream()).unwrap();
            }
            self.stream.synchronize().unwrap();
        }
    }
}

/// ページ固定メモリに写す。解放はしない (プロセスの終わりまで使う)
fn pin(src: &[u8]) -> &'static [u8] {
    // SAFETY: 確保した領域に src を写し、その領域を指すスライスを返す
    unsafe {
        let p = result::malloc_host(src.len(), 0).expect("ページ固定メモリを確保できない") as *mut u8;
        par_copy(src, p);
        std::slice::from_raw_parts(p, src.len())
    }
}

/// 重みを全コアで写す。mmap のページフォルトも各スレッドに分かれる
///
/// # Safety
/// dst は src.len() バイト書き込めて、src と重ならない
unsafe fn par_copy(src: &[u8], dst: *mut u8) {
    if src.len() < 1 << 20 {
        // SAFETY: 呼び出し側の約束どおり
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
        return;
    }
    let n = std::thread::available_parallelism().map_or(1, |n| n.get());
    let (size, dst) = (src.len().div_ceil(n), dst as usize);
    std::thread::scope(|s| {
        for (i, c) in src.chunks(size).enumerate() {
            // SAFETY: 各スレッドは dst の重ならない範囲に書く
            s.spawn(move || unsafe { std::ptr::copy_nonoverlapping(c.as_ptr(), (dst + i * size) as *mut u8, c.len()) });
        }
    });
}

mod draft;
mod host_attn;
pub use draft::{DRAFT_CTX, Drafter};

/// GPU から直接読み書きできる、ページ固定の主記憶を確保する。(GPU 側のアドレス, 主記憶側のアドレス)。解放はしない
fn mapped(bytes: usize) -> (P, usize) {
    // SAFETY: 確保に失敗したら止まる。DEVICEMAP で確保した領域の GPU 側のアドレスを得る
    unsafe {
        let h = result::malloc_host(bytes, sys::CU_MEMHOSTALLOC_DEVICEMAP).unwrap_or_else(|e| panic!("主記憶を {bytes} バイト固定できない: {e}"));
        let mut d: sys::CUdeviceptr = 0;
        sys::cuMemHostGetDevicePointer_v2(&mut d, h, 0).result().unwrap();
        (d, h as usize)
    }
}
