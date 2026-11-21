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

/// global 層の KV キャッシュの長さ。KV は f16 で、8192 トークンで global 層 268 MB、SWA 層のリング 250 MB。
/// f32 のときは 2048 が上限だった (8192 にすると VRAM の予算を超え、Windows が GPU メモリを退避して遅くなった)。
/// attn_part は分割1つ分のスコア (MAX_CTX / 16 個) を共有メモリに置く。
/// ponytail: 文脈長の上限。伸ばすなら KV を 8 ビットにするか、SWA 層のリングだけ残して global 層の KV を主記憶に置く
pub const MAX_CTX: usize = 8192;

/// prefill で一度に処理するトークン数の上限
pub const TMAX: usize = 512;

/// SWA 層の KV のリングバッファの行数。バッチの全トークンを書いてから注意を計算するので、窓幅に TMAX を足す
const RING: usize = EXTENT + TMAX;

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
    hist: [P; 4], // k, v, attn 出力, mlp 出力
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
    // 1トークン分の GPU の処理を、CPU で MoE を計算する所で区切って記録した CUDA Graph
    graphs: RefCell<Vec<CudaGraph>>,
    capturing: Cell<bool>,
    // prefill で、20〜23層の MXFP8 expert の一部を PCIe で送って GPU で計算するための送信用ストリームと置き場
    copy: Arc<CudaStream>,
    staging: [P; 2],
    up_done: [CudaEvent; 2],
    comp_done: [CudaEvent; 2],
}

/// prefill で1回に PCIe で送る MXFP8 expert の数。置き場は2つあり、片方を送るあいだにもう片方で計算する
const GROUP: usize = 8;
// MXFP8 の expert 1個あたりのバイト数
const W13: usize = 2 * FF * D;
const S13: usize = W13 / 32;
const W2: usize = D * FF;
const S2: usize = W2 / 32;

/// GPU と CPU の分担を釣り合わせるための見積もり (秒)。expert 8個 (66 MB) を PCIe で送る時間と、CPU で割り当て1つを計算する時間。
/// 実測: PCIe 4.0 x8 で 14.3 GB/s、CPU は 512 トークンの1層 (4096 割り当て) で約 130 ms。別の機械では測り直す
const GPU_PER_GROUP: f64 = 4.6e-3;
const CPU_PER_ASSIGN: f64 = 32e-6;

const KERNELS: [&str; 22] = [
    "mv_bf16", "gemm_bf16", "rmsnorm", "mix", "norm2", "control", "conv", "qkv_post", "attn_part", "attn_combine", "attn_tile", "add", "softcap", "topk", "moe_up_nvfp4", "moe_down_nvfp4",
    "moe_sort", "gemm_nvfp4_grouped", "situ_rows", "moe_sum", "gemm_mxfp8_grouped", "situ_sel",
];

fn as_bytes<T>(v: &[T]) -> &[u8] {
    // SAFETY: u16/f32/u8 の配列をバイト列として読むだけ
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}

fn f4(n: usize) -> u64 {
    (n * 4) as u64
}

impl Gpu {
    pub fn new(mut cpu: Model) -> Self {
        let ctx = CudaContext::new(0).expect("GPU を開けない");
        ctx.bind_to_thread().unwrap();
        // 20〜23層の MXFP8 expert (4.2 GB) をページ固定メモリに写す。prefill で PCIe から非同期に送るため。
        // CPU の計算もこちらを読むので、mmap した元のページは OS が追い出せる
        for l in cpu.layers.iter_mut().filter(|l| l.experts.mx) {
            let e = &mut l.experts;
            (e.w13, e.s13, e.w2, e.s2) = (pin(e.w13), pin(e.s13), pin(e.w2), pin(e.s2));
        }
        let cpu: &'static Model = Box::leak(Box::new(cpu));
        let src = format!("#define MAX_CTX_DEV {MAX_CTX}\n{}", include_str!("gpu.cu"));
        let ptx = compile_ptx_with_opts(src, CompileOptions { arch: Some("sm_120"), ..Default::default() })
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
        let slots = (0..N_SLOT)
            .map(|s| {
                let rows = if GLOBAL_LAYERS.contains(&phys(s)) { MAX_CTX } else { RING };
                GSlot {
                    kc: up.zeros(rows * 1024 * 2),
                    vc: up.zeros(rows * 1024 * 2),
                    hist: [1024, 1024, D, D].map(|ch| up.zeros(ch * 3 * 4)),
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
            part: fz(24 * 16 * 130, &mut up),
        };
        let unembed = up.bytes(as_bytes(cpu.unembed));
        let norm = f32s(&model::bf_vec(cpu.norm), &mut up);
        let output_query = f32s(cpu.output_query, &mut up);
        let staging = [0; 2].map(|_| up.zeros(GROUP * (W13 + S13 + W2 + S2)));
        let (free, _) = result::mem_get_info().unwrap();
        eprintln!("GPU に {:.2} GB 載せた。残り {:.2} GB", up.total as f64 / 1e9, free as f64 / 1e9);
        let ev = || ctx.new_event(None).unwrap();
        Self {
            cpu,
            copy: ctx.new_stream().unwrap(),
            staging,
            up_done: [ev(), ev()],
            comp_done: [ev(), ev()],
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
            graphs: RefCell::new(vec![]),
            capturing: Cell::new(false),
        }
    }

    fn launch(&self, name: &str, grid: (u32, u32), block: u32, smem: u32, args: &[u64]) {
        let mut vals = args.to_vec();
        let mut ptrs: Vec<*mut c_void> = vals.iter_mut().map(|v| v as *mut u64 as *mut c_void).collect();
        // SAFETY: 引数の数と型は gpu.cu の宣言に合わせてある。ポインタはどれも 64bit
        unsafe { result::launch_kernel(self.f[name].cu_function(), (grid.0, grid.1, 1), (block, 1, 1), smem, self.stream.cu_stream(), &mut ptrs) }
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

    /// Y[t] = W X[t]。T = 1 は行列×ベクトル、それより多ければタイル分割した行列積
    fn gemm(&self, w: P, x: P, y: P, rows: usize, cols: usize, t: usize) {
        if t == 1 {
            k!(self, "mv_bf16", (rows * 32).div_ceil(256), 256, 0; w, x, y, rows as i32, cols as i32);
        } else {
            k!(self, "gemm_bf16", (rows.div_ceil(64), t.div_ceil(32)), 128, 0; w, x, y, rows as i32, cols as i32, t as i32);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn mix(&self, blocks: P, nb: usize, cur: P, queries: P, nq: usize, out: P, t: usize) {
        k!(self, "mix", (nq, t), 256, 0; blocks, nb as i32, cur, queries, out, t as i32);
    }

    fn control(&self, cur: P, msg: P, c: &GCtrl, t: usize) {
        let b = &self.b;
        k!(self, "norm2", (2, t), 256, 0; cur, msg, b.cat);
        self.gemm(c.down, b.cat, b.g, 64, 2 * D, t);
        self.gemm(c.gf, b.g, b.gf, 2 * D, 64, t);
        k!(self, "control", (D.div_ceil(256), t), 256, 0; cur, msg, b.gf);
    }

    #[allow(clippy::too_many_arguments)]
    fn conv(&self, cur: P, cs: usize, w: P, dyn_: Option<usize>, hist: P, out: P, os: usize, ch: usize, cph: usize, t: usize) {
        let d = dyn_.map_or(-1, |d| d as i32);
        k!(self, "conv", ch.div_ceil(256), 256, 0; cur, cs as i32, w, d, hist, out, os as i32, ch as i32, cph as i32, t as i32);
    }

    /// トークンの埋め込みと先頭の位置を GPU に送る
    fn inputs(&self, tokens: &[u32]) {
        assert!(self.pos + tokens.len() <= MAX_CTX, "文脈長の上限 {MAX_CTX} を超えた");
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

    /// 1トークンを処理して、語彙の logits を返す。
    /// 最初のトークンでカーネル列を CUDA Graph に記録し、以降は再生する。トレース中は記録せず毎回カーネルを投げる。
    pub fn forward(&mut self, token: u32) -> Vec<f32> {
        self.inputs(&[token]);
        if self.graphs.borrow().is_empty() || model::tracing() {
            self.capturing.set(!model::tracing());
            self.begin();
            self.record(1);
            self.end();
            self.capturing.set(false);
        } else {
            for (i, g) in self.graphs.borrow().iter().enumerate() {
                g.launch().unwrap();
                if i < N_LAYER - MXFP8_FROM {
                    self.cpu_moe(MXFP8_FROM + i);
                }
            }
        }
        self.pos += 1;
        self.dtoh(self.b.logits, VOCAB)
    }

    /// プロンプトを TMAX トークンずつまとめて処理し、最後のトークンの logits を返す。
    pub fn prefill(&mut self, tokens: &[u32]) -> Vec<f32> {
        for chunk in tokens.chunks(TMAX) {
            self.inputs(chunk);
            self.record(chunk.len());
            self.pos += chunk.len();
        }
        self.dtoh(self.b.logits, VOCAB)
    }

    /// 次に処理するトークンの位置 (= ここまでに処理したトークン数)
    pub fn pos(&self) -> usize {
        self.pos
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
            self.graphs.borrow_mut().push(g);
        }
    }

    /// (生成) 20〜23層の MoE を CPU で計算する。b.h を読み b.y に書く
    fn cpu_moe(&self, il: usize) {
        let h = self.dtoh(self.b.h, D);
        let y = self.cpu.moe(&self.cpu.layers[il], &h);
        self.htod(self.b.y, &y);
    }

    /// (prefill) 20〜23層の MoE を、expert 8個ずつの組に分け、一部の組は PCIe で送って GPU で、残りは CPU で同時に計算する。
    /// b.h を読み b.y に書く
    fn hybrid_moe(&self, il: usize, t: usize) {
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

    /// T トークン分のカーネル列を投げる。logits は最後のトークンの分だけ求める
    fn record(&self, t: usize) {
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
                    self.mix(b.bank, 1, b.bank + slab(1), self.layers[ip].msg_query, 4, b.msgs, t);
                }
                self.layer(ip, ip, slot, &mut nb, pass > 0, t);
            }
        }
        for il in CORE.1..N_LAYER {
            self.layer(il, il, il, &mut nb, false, t);
        }
        self.mix(b.blocks, nb, b.x, self.output_query, 1, b.h, t);
        let last = b.h + f4((t - 1) * D);
        k!(self, "rmsnorm", 1, 256, 0; last, self.norm, last, D as i32, 0i32, 1i32);
        self.gemm(self.unembed, last, b.logits, 201024, D, 1);
        k!(self, "softcap", VOCAB.div_ceil(256), 256, 0; b.logits, VOCAB as i32);
    }

    #[allow(clippy::too_many_arguments)]
    fn layer(&self, il: usize, ip: usize, slot: usize, nb: &mut usize, msgs: bool, t: usize) {
        let (l, s, b) = (&self.layers[il], &self.slots[slot], &self.b);
        let slab = |i: usize| f4(i * t * D);
        let ti = t as i32;
        // attention の入力: SWA は1本、global は Q, K, V に1本ずつ
        if il > 0 {
            self.mix(b.blocks, *nb, b.x, l.attn_query, if l.global { 3 } else { 1 }, b.inp, t);
        } else {
            self.copy(b.inp, b.x, t * D);
        }
        let separate = l.global || msgs;
        if !l.global && msgs {
            self.copy(b.inp + slab(1), b.inp, t * D);
            self.copy(b.inp + slab(2), b.inp, t * D);
        }
        if msgs {
            for j in 0..3 {
                self.control(b.inp + slab(j), b.msgs + slab(j), &l.ctrl[j], t);
            }
        }
        let n_in = if separate { 3 } else { 1 };
        k!(self, "rmsnorm", n_in * t, 256, 0; b.inp, l.norms, b.inp, D as i32, if l.global { D as i32 } else { 0i32 }, ti);
        let (xq, xk, xv) = if separate { (b.inp, b.inp + slab(1), b.inp + slab(2)) } else { (b.inp, b.inp, b.inp) };

        let nk = if l.global { 1024 + 32 } else { 1024 };
        self.gemm(l.wq, xq, b.qx, QX, D, t);
        self.gemm(l.wk, xk, b.kx, nk, D, t);
        self.gemm(l.wv, xv, b.vx, nk, D, t);
        let dyn_ = l.global.then_some(1024);
        self.conv(b.kx, nk, l.k_conv, dyn_, s.hist[0], b.kp, 1024, 1024, 128, t);
        self.conv(b.vx, nk, l.v_conv, dyn_, s.hist[1], b.vp, 1024, 1024, 128, t);
        let ring = if l.global { 0 } else { RING as i32 };
        k!(self, "qkv_post", (64, t), 128, 0; b.qx, b.kp, b.vp, l.q_clip, l.rel_clip, b.pos, ring, s.kc, s.vc);
        if t == 1 {
            k!(self, "attn_part", (24, 16), 128, 0; b.qx, s.kc, s.vc, l.rel_proj, b.part, b.pos, ring);
            k!(self, "attn_combine", 24, 128, 0; b.qx, b.part, b.o);
        } else {
            k!(self, "attn_tile", (24, t.div_ceil(32)), 128, 0; b.qx, s.kc, s.vc, l.rel_proj, b.o, b.pos, ring, ti);
        }
        self.gemm(l.wo, b.o, b.a, D, 3072, t);
        self.conv(b.a, D, l.attn_conv, None, s.hist[2], b.a, D, D, D, t);
        // ブロックの先頭層は残差を引き継がず、入力をブロック列に積む
        if ip % BLOCK == 0 {
            self.copy(b.blocks + slab(*nb), b.x, t * D);
            *nb += 1;
            self.copy(b.x, b.a, t * D);
        } else {
            k!(self, "add", (t * D).div_ceil(256), 256, 0; b.x, b.a, b.x, (t * D) as i32);
        }

        self.mix(b.blocks, *nb, b.x, l.mlp_query, 1, b.h, t);
        if msgs {
            self.control(b.h, b.msgs + slab(3), &l.ctrl[3], t);
        }
        k!(self, "rmsnorm", t, 256, 0; b.h, l.mlp_norm, b.h, D as i32, 0i32, ti);
        match l.experts {
            Some([w13, s13, a13, w2, s2, a2]) => {
                self.gemm(l.router, b.h, b.rl, 128, D, t);
                k!(self, "topk", t, 128, 0; b.rl, l.router_bias, b.ids, b.wts);
                if t == 1 {
                    k!(self, "moe_up_nvfp4", (TOP_K * FF * 32).div_ceil(256), 256, 0; w13, s13, a13, b.ids, b.h, b.act);
                    k!(self, "moe_down_nvfp4", (D * 32).div_ceil(256), 256, 0; w2, s2, a2, b.ids, b.wts, b.act, b.y);
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
            None if t == 1 => {
                self.end();
                self.cpu_moe(il);
                self.begin();
            }
            None => self.hybrid_moe(il, t),
        }
        self.conv(b.y, D, l.mlp_conv, None, s.hist[3], b.y, D, D, D, t);
        k!(self, "add", (t * D).div_ceil(256), 256, 0; b.x, b.y, b.x, (t * D) as i32);
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
        for (i, c) in src.chunks(PIN).enumerate() {
            // SAFETY: pin は PIN バイトある。前の転送が終わってから上書きする
            unsafe {
                std::ptr::copy_nonoverlapping(c.as_ptr(), self.pin, c.len());
                let staged = std::slice::from_raw_parts(self.pin, c.len());
                result::memcpy_htod_async(p + (i * PIN) as u64, staged, self.stream.cu_stream()).unwrap();
            }
            self.stream.synchronize().unwrap();
        }
        p
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
