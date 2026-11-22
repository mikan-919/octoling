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
    experts: Option<[P; 6]>, // w13, s13, a13, w2, s2, a2。MXFP8 の層は None で CPU が計算する
}

struct GSlot {
    kc: P,
    vc: P,
    host: Option<[usize; 2]>, // KV が主記憶にあれば、その K と V の主記憶側のアドレス
    hist: [P; 4], // k, v, attn 出力, mlp 出力
    seq: [P; 4],  // 検証のバッチの [直前の入力3つ; 入力 SPEC_T 個]。外れた下書きを捨てるとき hist を戻す
}

// TMAX トークン分の作業領域。並びは gpu.cu の先頭に書いたとおり [種類][T][要素]
struct Bufs {
    x: P,
    pos: P,    // i32。バッチの先頭トークンの位置。カーネルはここから読む
    blocks: P, // [8][T][D]
    prelude: P,
    bank: P, // [2][T][D]
    msgs: P, // [4][T][D]
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
    taps: P,       // [4][TMAX][D]。下書きモデルが読む、スロット 4, 28, 46 の層出力と最後の正規化の出力
    amax: P,       // [SPEC_T] i32。検証のバッチの各行の、最大の logits の位置
    logits_all: P, // [SPEC_T][201024]
    acc: P,        // [TMAX][24][HD + 2]。主記憶の KV の塊ごとに足し込む注意の途中結果
    done: P,       // u32 [SPEC_T * 24 + 1]。最後に終わったブロックを決める数 (attn_part のヘッドごと、router_topk)。使い終わると 0 に戻る
    best: P,       // u64。最後の行の最大の logits の (値, 位置) を並べたもの (softcap が書く)
}

const QX: usize = 3072 + 384 + 3072 + 24;

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
    b: Bufs,
    pos: usize,
    ctx: usize,
    // T トークン分の GPU の処理を、CPU で MoE を計算する所で区切って記録した CUDA Graph。(T, 検証か) ごと
    graphs: RefCell<HashMap<(usize, bool), Vec<CudaGraph>>>,
    recording: RefCell<Vec<CudaGraph>>,
    rewind_tab: P, // conv_rewind に渡す (seq, hist, ch) の表
    capturing: Cell<bool>,
    // prefill で、20〜23層の MXFP8 expert の一部を PCIe で送って GPU で計算するための送信用ストリームと置き場
    copy: Arc<CudaStream>,
    staging: [P; 2],
    up_done: [CudaEvent; 2],
    comp_done: [CudaEvent; 2],
    xc: RefCell<XCache>,
    fill_done: CudaEvent, // 枠への転送の終わり
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
const VRAM_MARGIN: usize = 400 << 20;

/// 生成で、20〜23層 (MXFP8) のよく選ばれる expert を VRAM に置いておく枠。選ばれた expert が枠にあれば GPU で、なければ CPU で計算する。
/// 枠 j は base + j * XSLOT。先頭の 2 * GROUP 枠は prefill の置き場 (staging) と同じ場所なので、prefill で使ったら空にする
struct XCache {
    base: P,
    owner: Vec<Option<(usize, usize)>>, // 枠 → (層 - 20, expert)
    slot: [[Option<usize>; N_EXP]; 4],
    cnt: [[f32; N_EXP]; 4], // 選ばれた回数 (XDECAY で減らす)
    pending: Vec<(usize, usize, usize)>, // このトークンで入れ替えると決めた (層 - 20, expert, 枠)。23層のあとにまとめて送る
}

/// GPU と CPU の分担を釣り合わせるための見積もり (秒)。expert 8個 (66 MB) を PCIe で送る時間と、CPU で割り当て1つを計算する時間。
/// 実測: PCIe 4.0 x8 で 14.3 GB/s、CPU は 512 トークンの1層 (4096 割り当て) で約 130 ms。別の機械では測り直す
const GPU_PER_GROUP: f64 = 4.6e-3;
const CPU_PER_ASSIGN: f64 = 32e-6;

const KERNELS: [&str; 44] = [
    "mv_bf16", "gemm_bf16", "rmsnorm", "mix", "norm2", "control", "conv", "qkv_post", "attn_part", "attn_tile", "add", "softcap", "topk", "moe_up_nvfp4", "moe_down_nvfp4",
    "moe_sort", "gemm_nvfp4_grouped", "situ_rows", "moe_sum", "gemm_mxfp8_grouped", "situ_sel", "conv_rewind", "argmax_rows",
    "quant_rows_bf16", "quant_cols_f32", "mv_fp8", "tapmix", "dr_ctx_store", "dr_qk", "dr_attn", "silu_mul", "concat_frnn", "add_bias_rows", "fold_finish", "mv_bf16_t", "attn_blk", "attn_fin", "moe_up_mx", "moe_down_mx", "ctrl_down", "ctrl_up", "mv3_bf16", "qkv_conv_post", "router_topk",
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
                    experts: (!e.mx).then(|| [e.w13, e.s13, as_bytes(e.a13), e.w2, e.s2, as_bytes(e.a2)].map(|b| up.bytes(b))),
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
        let fz = |n: usize, up: &mut Uploader| up.zeros(n * 4);
        let t = TMAX;
        let b = Bufs {
            x: fz(t * D, &mut up),
            pos: fz(1, &mut up),
            blocks: fz(8 * t * D, &mut up),
            prelude: fz(t * D, &mut up),
            bank: fz(2 * t * D, &mut up),
            msgs: fz(4 * t * D, &mut up),
            inp: fz(3 * t * D, &mut up),
            qx: fz(t * QX, &mut up),
            kx: fz(t * (1024 + 32), &mut up),
            vx: fz(t * (1024 + 32), &mut up),
            kp: fz(t * 1024, &mut up),
            vp: fz(t * 1024, &mut up),
            o: fz(t * 3072, &mut up),
            a: fz(t * D, &mut up),
            h: fz(t * D, &mut up),
            y: fz(t * D, &mut up),
            cat: fz(t * 2 * D, &mut up),
            g: fz(t * 64, &mut up),
            gf: fz(t * 2 * D, &mut up),
            rl: fz(t * 128, &mut up),
            ids: fz(t * TOP_K, &mut up),
            wts: fz(t * TOP_K, &mut up),
            act: fz(t * TOP_K * FF, &mut up),
            gu: fz(t * TOP_K * 2 * FF, &mut up),
            yd: fz(t * TOP_K * D, &mut up),
            order: fz(t * TOP_K, &mut up),
            off: fz(129, &mut up),
            logits: fz(201024, &mut up),
            part: fz(SPEC_T * 24 * 16 * 130, &mut up),
            taps: fz(4 * t * D, &mut up),
            amax: fz(SPEC_T, &mut up),
            logits_all: fz(SPEC_T * 201024, &mut up),
            acc: if n_ctx > VRAM_CTX { fz(t * 24 * 130, &mut up) } else { 0 },
            done: fz(SPEC_T * 24 + 1, &mut up),
            best: fz(2, &mut up),
        };
        let unembed = up.bytes(as_bytes(cpu.unembed));
        let norm = f32s(&model::bf_vec(cpu.norm), &mut up);
        let output_query = f32s(cpu.output_query, &mut up);
        // 残りの VRAM で expert の枠をとる。OCTO_EXPERT_SLOTS で数を決められる (prefill の置き場を兼ねるので 2 * GROUP 以上)
        let (free, _) = result::mem_get_info().unwrap();
        let n_slots = std::env::var("OCTO_EXPERT_SLOTS")
            .map_or(free.saturating_sub(VRAM_MARGIN) / XSLOT, |s| s.parse().expect("OCTO_EXPERT_SLOTS は数"))
            .clamp(2 * GROUP, 255);
        let xbase = up.zeros(n_slots * XSLOT);
        let staging = [xbase, xbase + (GROUP * XSLOT) as u64];
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
            xc: RefCell::new(XCache { base: xbase, owner: vec![None; n_slots], slot: [[None; N_EXP]; 4], cnt: [[0.0; N_EXP]; 4], pending: vec![] }),
            fill_done: ev(),
            _ctx: ctx.clone(),
            stream,
            _module: module,
            f,
            layers,
            slots,
            unembed,
            norm,
            output_query,
            b,
            pos: 0,
            ctx: n_ctx,
            graphs: RefCell::new(HashMap::new()),
            recording: RefCell::new(vec![]),
            rewind_tab,
            capturing: Cell::new(false),
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
    fn mix(&self, blocks: P, nb: usize, cur: P, queries: P, nq: usize, out: P, t: usize, norm: Option<(P, usize)>) {
        assert!(nb < 5, "mix は状態 5 個まで (MIX_NS)");
        let (nw, nws) = norm.unwrap_or((0, 0));
        k!(self, "mix", (nq, t), 256, 0; blocks, nb as i32, cur, queries, out, t as i32, nw, nws as i32);
    }

    /// control cs[j] (j < 3) を、入力 j に メッセージ msg + slab(j) でかけて out + slab(j) に書く。並びは [j][T][D]。
    /// 入力 j は shared なら cur (全部共通)、そうでなければ cur + slab(j) で、out と同じでよい。T = 1 なら全部を1組の起動で計算する
    fn control(&self, cur: P, shared: bool, out: P, msg: P, cs: &[GCtrl], t: usize) {
        let b = &self.b;
        let slab = |j: usize| f4(j * t * D);
        if t == 1 {
            let w = |j: usize, f: fn(&GCtrl) -> P| cs.get(j).map_or(0, f);
            let stride = if shared { 0 } else { D as i32 };
            k!(self, "ctrl_down", (8, cs.len()), 256, 0; cur, stride, msg, w(0, |c| c.down), w(1, |c| c.down), w(2, |c| c.down), b.g);
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
        self.htod(self.b.x, &e);
        self.htod(self.b.pos, &[self.pos as i32]);
    }

    /// 位置 self.pos から T トークンを処理する。T <= SPEC_T なら、最初の1回でカーネル列を CUDA Graph に記録し、
    /// 以降は再生する。それより多いときとトレース中は、毎回カーネルを投げる。
    /// verify なら全行の最大の logits の位置を b.amax に、そうでなければ最後の行の logits を b.logits に書く
    fn run(&self, t: usize, verify: bool) {
        let key = (t, verify);
        let small = t <= SPEC_T && !model::tracing();
        if small && let Some(gs) = self.graphs.borrow().get(&key) {
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
                if i < N_LAYER - MXFP8_FROM {
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
        self.capturing.set(small);
        self.begin();
        self.record(t, verify);
        self.end();
        self.capturing.set(false);
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
        self.dtoh(self.b.logits, VOCAB)
    }

    /// 1トークンを処理して、logits の最大の位置 (貪欲法の次のトークン) を返す。logits は GPU で調べ、位置だけを読む
    pub fn forward_greedy(&mut self, token: u32) -> u32 {
        self.inputs(&[token]);
        self.run(1, false);
        self.pos += 1;
        0xffff_ffff - self.dtoh::<u64>(self.b.best, 1)[0] as u32
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
        self.dtoh(self.b.logits, VOCAB)
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
        self.dtoh::<i32>(self.b.amax, tokens.len()).into_iter().map(|t| t as u32).collect()
    }

    /// 直前の verify の t トークンのうち、先頭 keep 個だけを残す。畳み込みの状態を戻し、位置を下げる。
    /// KV キャッシュは位置で読むので、捨てた位置は次に書くときに上書きされる
    pub fn rewind(&mut self, keep: usize, t: usize) {
        if keep < t {
            k!(self, "conv_rewind", (D.div_ceil(256), self.slots.len() * 4), 256, 0; self.rewind_tab, keep as i32);
            self.pos -= t - keep;
        }
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
        let h: Vec<f32> = self.dtoh(self.b.h, t * D);
        let l = &self.cpu.layers[il];
        if t == 1 {
            return self.split_moe(il, l, &h);
        }
        let routes: Vec<_> = h.chunks(D).map(|x| self.cpu.route(l, x)).collect();
        let ids: Vec<i32> = routes.iter().flat_map(|r| r.0.map(|e| e as i32)).collect();
        let wts: Vec<f32> = routes.iter().flat_map(|r| r.1).collect();
        let y = self.cpu.moe_batch(l, &h, &ids, &wts, |_| true);
        self.htod(self.b.y, &y);
    }

    /// (生成) 20〜23層の MoE。選ばれた expert のうち枠にあるものを GPU で、残りを同時に CPU で計算する。h は b.h の写しで、b.y に書く。
    /// そのあと、CPU で計算した expert のうち選ばれた回数の多いものを、空いた枠か回数の最も少ない枠へ PCIe で送っておく
    fn split_moe(&self, il: usize, l: &Layer, h: &[f32]) {
        let (li, b) = (il - MXFP8_FROM, &self.b);
        let (sel, wts) = self.cpu.route(l, h);
        let mut xc = self.xc.borrow_mut();
        if li == 0 {
            xc.cnt.iter_mut().flatten().for_each(|c| *c *= XDECAY);
        }
        for &e in &sel {
            xc.cnt[li][e] += 1.0;
        }
        let (mut slots, mut g, mut n) = (0u64, [0.0f32; TOP_K], 0);
        let (mut on_cpu, mut cpu_wts) = (vec![], vec![]);
        for (&e, &w) in sel.iter().zip(&wts) {
            match xc.slot[li][e] {
                Some(s) => {
                    slots |= (s as u64) << (8 * n);
                    g[n] = w;
                    n += 1;
                }
                None => {
                    on_cpu.push(e);
                    cpu_wts.push(w);
                }
            }
        }
        if n > 0 {
            // 前のトークンの終わりに送った枠が届いてから読む
            self.stream.wait(&self.fill_done).unwrap();
            k!(self, "moe_up_mx", (n * FF * 32).div_ceil(256), 256, 0; xc.base, n as i32, slots, b.h, b.act);
            k!(self, "moe_down_mx", (D * 32).div_ceil(256), 256, 0;
                xc.base, n as i32, slots, b.act, g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7], b.yd);
        }
        let y = if on_cpu.is_empty() { vec![0.0; D] } else { self.cpu.moe_some(l, h, &on_cpu, &cpu_wts) };
        if n > 0 {
            self.htod(b.a, &y);
            k!(self, "add", D.div_ceil(256), 256, 0; b.yd, b.a, b.y, D as i32);
        } else {
            self.htod(b.y, &y);
        }

        // 入れ替える枠を決める。転送は、CPU の計算と主記憶の帯域を取り合わないよう、23層のあと (CPU が空く GPU の区間) にまとめて出す
        on_cpu.sort_by(|&a, &c| xc.cnt[li][c].total_cmp(&xc.cnt[li][a]));
        for &ex in on_cpu.iter().take(XFILLS) {
            let score = |xc: &XCache, s: usize| xc.owner[s].map_or(-1.0, |(l2, e2)| xc.cnt[l2][e2]);
            let victim = (0..xc.owner.len()).min_by(|&a, &c| score(&xc, a).total_cmp(&score(&xc, c))).unwrap();
            if score(&xc, victim) >= xc.cnt[li][ex] {
                break;
            }
            if let Some((l2, e2)) = xc.owner[victim] {
                xc.slot[l2][e2] = None;
            }
            xc.owner[victim] = Some((li, ex));
            xc.slot[li][ex] = Some(victim);
            xc.pending.push((li, ex, victim));
        }
        if il == N_LAYER - 1 {
            // htod で同期したので、GPU はもう枠を読んでいない
            for (li, ex, victim) in std::mem::take(&mut xc.pending) {
                let e = &self.cpu.layers[MXFP8_FROM + li].experts;
                let mut at = xc.base + (victim * XSLOT) as u64;
                for (src, size) in [(e.w13, W13), (e.s13, S13), (e.w2, W2), (e.s2, S2)] {
                    // SAFETY: src はページ固定メモリ。枠は XSLOT バイトある
                    unsafe { result::memcpy_htod_async(at, &src[ex * size..(ex + 1) * size], self.copy.cu_stream()) }.unwrap();
                    at += size as u64;
                }
            }
            self.fill_done.record(&self.copy).unwrap();
        }
    }

    /// prefill で置き場 (expert の先頭の 2 * GROUP 枠) を使う前に呼ぶ。枠への転送が終わるのを待ち、その枠を空にする
    fn take_staging(&self) {
        self.stream.wait(&self.fill_done).unwrap();
        let mut xc = self.xc.borrow_mut();
        for s in 0..2 * GROUP {
            if let Some((l2, e2)) = xc.owner[s].take() {
                xc.slot[l2][e2] = None;
            }
        }
    }


    /// (prefill) 20〜23層の MoE を、expert 8個ずつの組に分け、一部の組は PCIe で送って GPU で、残りは CPU で同時に計算する。
    /// b.h を読み b.y に書く
    fn hybrid_moe(&self, il: usize, t: usize) {
        self.take_staging();
        let (b, l, cl) = (&self.b, &self.layers[il], &self.cpu.layers[il]);
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
        let b = &self.b;
        let slab = |i: usize| f4(i * t * D); // [種類][T][D] の i 番目
        let mut nb = 0;
        for il in 0..CORE.0 {
            self.layer(il, il, il, &mut nb, false, t);
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
                if pass > 0 {
                    self.mix(b.bank, 1, b.bank + slab(1), self.layers[ip].msg_query, 4, b.msgs, t, None);
                }
                self.layer(ip, ip, slot, &mut nb, pass > 0, t);
            }
        }
        for il in CORE.1..N_LAYER {
            self.layer(il, il, il, &mut nb, false, t);
        }
        self.mix(b.blocks, nb, b.x, self.output_query, 1, b.h, t, None);
        // 正規化した最後の隠れ状態は、下書きモデルの4つ目の入力でもある
        let normed = b.taps + f4(3 * TMAX * D);
        k!(self, "rmsnorm", t, 256, 0; b.h, self.norm, normed, 0i32, t as i32);
        if verify {
            // softcap は単調なので、最大の位置は変わらない
            self.gemm(self.unembed, normed, b.logits_all, 201024, D, t);
            k!(self, "argmax_rows", t, 1024, 0; b.logits_all, 201024i32, VOCAB as i32, b.amax);
        } else {
            self.gemm(self.unembed, normed + f4((t - 1) * D), b.logits, 201024, D, 1);
            // SAFETY: best は 8 バイトある
            unsafe { result::memset_d8_async(b.best, 0, 8, self.stream.cu_stream()) }.unwrap();
            k!(self, "softcap", VOCAB.div_ceil(256), 256, 0; b.logits, VOCAB as i32, b.best);
        }
    }

    /// (prefill、主記憶の KV) 位置 0..pos+t の KV を KB 位置ずつ置き場に送り、塊ごとに注意を足し込む。
    /// GPU が主記憶を直接読むと、クエリのタイルごとに同じ KV を読み直すことになり遅い
    fn attn_host(&self, host: [usize; 2], l: &GLayer, t: usize) {
        self.take_staging();
        // 置き場 1 つ (65.7 MB) に K と V が入る位置数 (1 位置 2 KB ずつ)。16 の倍数に切り下げる
        const KB: usize = GROUP * (W13 + S13 + W2 + S2) / 4096 / 16 * 16;
        let (b, st, end) = (&self.b, self.staging[0], self.pos + t);
        for (i, b0) in (0..end).step_by(KB).enumerate() {
            let nb = KB.min(end - b0);
            for (j, &h) in host.iter().enumerate() {
                // SAFETY: host はページ固定の主記憶で、end 行以上ある。置き場は K と V で 2 * KB 行ある
                unsafe {
                    let src = std::slice::from_raw_parts((h as *const u16).add(b0 * 1024), nb * 1024);
                    result::memcpy_htod_async(st + (j * KB * 2048) as u64, src, self.stream.cu_stream()).unwrap();
                }
            }
            k!(self, "attn_blk", (24, t.div_ceil(32)), 128, 0;
                b.qx, st, st + (KB * 2048) as u64, b0 as i32, nb as i32, l.rel_proj, b.acc, b.pos, t as i32, (i == 0) as i32);
        }
        k!(self, "attn_fin", (24, t), 128, 0; b.qx, b.acc, b.o);
    }

    #[allow(clippy::too_many_arguments)]
    fn layer(&self, il: usize, ip: usize, slot: usize, nb: &mut usize, msgs: bool, t: usize) {
        let (l, s, b) = (&self.layers[il], &self.slots[slot], &self.b);
        let slab = |i: usize| f4(i * t * D);
        let ti = t as i32;
        // attention の入力: SWA は1本、global は Q, K, V に1本ずつ
        // control がなければ、正規化も mix のカーネルでかける
        let wstride = if l.global { D } else { 0 };
        let fused = il > 0 && !msgs;
        if il > 0 {
            self.mix(b.blocks, *nb, b.x, l.attn_query, if l.global { 3 } else { 1 }, b.inp, t, fused.then_some((l.norms, wstride)));
        } else {
            self.copy(b.inp, b.x, t * D);
        }
        let separate = l.global || msgs;
        if msgs {
            // SWA は1本の入力から Q, K, V の3本を作る
            self.control(b.inp, !l.global, b.inp, b.msgs, &l.ctrl[..3], t);
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
                s.hist[0], s.hist[1], s.seq[0], s.seq[1], l.q_clip, l.rel_clip, b.pos, ring, s.kc, s.vc);
        } else {
            self.conv(b.kx, nk, l.k_conv, dyn_, s.hist[0], b.kp, 1024, 1024, 128, t, seq(0), 0);
            self.conv(b.vx, nk, l.v_conv, dyn_, s.hist[1], b.vp, 1024, 1024, 128, t, seq(1), 0);
            k!(self, "qkv_post", (64, t), 128, 0; b.qx, b.kp, b.vp, l.q_clip, l.rel_clip, b.pos, ring, s.kc, s.vc);
        }
        if t <= SPEC_T {
            // 主記憶の KV も GPU から直接読む
            k!(self, "attn_part", (24, 16 * t), 128, 0; b.qx, s.kc, s.vc, l.rel_proj, b.part, b.pos, ring, b.o, b.done);
        } else if let Some(host) = s.host {
            self.attn_host(host, l, t);
        } else {
            k!(self, "attn_tile", (24, t.div_ceil(32)), 128, 0; b.qx, s.kc, s.vc, l.rel_proj, b.o, b.pos, ring, ti);
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
        self.mix(b.blocks, *nb, b.x, l.mlp_query, 1, b.h, t, mlp_fused.then_some((l.mlp_norm, 0)));
        if msgs {
            self.control(b.h, false, b.h, b.msgs + slab(3), &l.ctrl[3..], t);
        }
        if !mlp_fused {
            k!(self, "rmsnorm", t, 256, 0; b.h, l.mlp_norm, b.h, 0i32, ti);
        }
        match l.experts {
            Some([w13, s13, a13, w2, s2, a2]) => {
                if t == 1 {
                    k!(self, "router_topk", 16, 256, 0; l.router, b.h, b.rl, l.router_bias, b.ids, b.wts, b.done + f4(SPEC_T * 24));
                } else {
                    self.gemm(l.router, b.h, b.rl, 128, D, t);
                    k!(self, "topk", t, 128, 0; b.rl, l.router_bias, b.ids, b.wts);
                }
                if t <= SPEC_T {
                    k!(self, "moe_up_nvfp4", (t * TOP_K * FF * 32).div_ceil(256), 256, 0; w13, s13, a13, b.ids, b.h, b.act, ti);
                    k!(self, "moe_down_nvfp4", (t * D * 32).div_ceil(256), 256, 0; w2, s2, a2, b.ids, b.wts, b.act, b.y, ti);
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
            None if t <= SPEC_T => {
                self.end();
                self.cpu_moe(il, t);
                self.begin();
            }
            None => self.hybrid_moe(il, t),
        }
        self.conv(b.y, D, l.mlp_conv, None, s.hist[3], b.y, D, D, D, t, seq(3), b.x);
        if let Some(j) = TAP_SLOTS.iter().position(|&s| s == slot) {
            self.copy(b.taps + f4(j * TMAX * D), b.x, t * D);
        }
        if model::tracing() && t == 1 {
            model::trace(slot, &self.dtoh(b.x, D));
        }
    }
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

    fn zeros(&mut self, n: usize) -> P {
        // SAFETY: 確保して 0 で埋める。解放はしない
        let p = unsafe { result::malloc_sync(n.max(4)) }.unwrap_or_else(|e| panic!("VRAM を {n} バイト確保できない: {e}"));
        unsafe { result::memset_d8_async(p, 0, n.max(4), self.stream.cu_stream()) }.unwrap();
        self.total += n;
        p
    }

    fn bytes(&mut self, src: &[u8]) -> P {
        let p = self.zeros(src.len());
        self.to(p, src);
        p
    }

    /// 確保済みの dst に送る。送り終わるまで待つ
    fn to(&mut self, dst: P, src: &[u8]) {
        for (i, c) in src.chunks(PIN).enumerate() {
            // SAFETY: pin は PIN バイトある。前の転送が終わってから上書きする
            unsafe {
                std::ptr::copy_nonoverlapping(c.as_ptr(), self.pin, c.len());
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
        std::ptr::copy_nonoverlapping(src.as_ptr(), p, src.len());
        std::slice::from_raw_parts(p, src.len())
    }
}

mod draft;
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
