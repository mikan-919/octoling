//! GPU (CUDA) 版の順伝播。計算の順序は model.rs の CPU 版と同じ。
//! 20〜23層の MXFP8 expert は VRAM に載らないので CPU で計算する。埋め込み表も CPU に置く。

use crate::model::{self, BLOCK, CORE, D, EXTENT, FF, GLOBAL_LAYERS, Layer, MXFP8_FROM, Model, N_LAYER, N_SLOT, PASSES, TOP_K, VOCAB, W};
use cudarc::driver::{CudaContext, CudaFunction, CudaGraph, CudaModule, CudaStream, result, sys};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;

/// global 層の KV キャッシュの長さ。8192 にすると VRAM の予算を超え、Windows が GPU メモリを退避して遅くなる。
/// attn カーネルはスコアを共有メモリに置くので、48KB に収まる長さにもする。
/// ponytail: 文脈長の上限。伸ばすなら KV を BF16 にするか、スコアを分割して求める
pub const MAX_CTX: usize = 2048;

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

macro_rules! k {
    ($g:expr, $name:literal, $grid:expr, $block:expr, $smem:expr; $($a:expr),* $(,)?) => {
        $g.launch($name, $grid as u32, $block as u32, $smem as u32, &[$(Arg::bits($a)),*])
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

// 1トークン分の作業領域
struct Bufs {
    x: P,
    pos: P, // i32。カーネルはここから位置を読む
    blocks: P, // [8][D]
    prelude: P,
    bank: P, // [2][D]
    msgs: P, // [4][D]
    inp: P,  // [3][D]
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
    act: P,
    logits: P,
}

const QX: usize = 3072 + 384 + 3072 + 24;
const R_OFF: usize = 3072;
const GATE_OFF: usize = 3072 + 384;
const SINK_OFF: usize = 3072 + 384 + 3072;

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
}

const KERNELS: [&str; 13] = ["mv_bf16", "rmsnorm", "mix", "norm2", "control", "conv", "qkv_post", "attn", "add", "softcap", "topk", "moe_up_nvfp4", "moe_down_nvfp4"];

fn as_bytes<T>(v: &[T]) -> &[u8] {
    // SAFETY: u16/f32/u8 の配列をバイト列として読むだけ
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}

impl Gpu {
    pub fn new(cpu: Model) -> Self {
        let cpu: &'static Model = Box::leak(Box::new(cpu));
        let ptx = compile_ptx_with_opts(include_str!("gpu.cu"), CompileOptions { arch: Some("sm_120"), ..Default::default() })
            .unwrap_or_else(|e| panic!("CUDA カーネルのコンパイルに失敗: {e}"));
        let ctx = CudaContext::new(0).expect("GPU を開けない");
        ctx.bind_to_thread().unwrap();
        // 既定のストリームは CUDA Graph に記録できない
        let stream = ctx.new_stream().unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let f = KERNELS.iter().map(|&n| (n, module.load_function(n).unwrap())).collect();

        let mut up = Uploader::new(&ctx, &stream);
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
                let rows = if GLOBAL_LAYERS.contains(&phys(s)) { MAX_CTX } else { EXTENT };
                GSlot {
                    kc: up.zeros(rows * 1024 * 4),
                    vc: up.zeros(rows * 1024 * 4),
                    hist: [1024, 1024, D, D].map(|ch| up.zeros(ch * 3 * 4)),
                }
            })
            .collect();
        let fz = |n: usize, up: &mut Uploader| up.zeros(n * 4);
        let b = Bufs {
            x: fz(D, &mut up),
            pos: fz(1, &mut up),
            blocks: fz(8 * D, &mut up),
            prelude: fz(D, &mut up),
            bank: fz(2 * D, &mut up),
            msgs: fz(4 * D, &mut up),
            inp: fz(3 * D, &mut up),
            qx: fz(QX, &mut up),
            kx: fz(1024 + 32, &mut up),
            vx: fz(1024 + 32, &mut up),
            kp: fz(1024, &mut up),
            vp: fz(1024, &mut up),
            o: fz(3072, &mut up),
            a: fz(D, &mut up),
            h: fz(D, &mut up),
            y: fz(D, &mut up),
            cat: fz(2 * D, &mut up),
            g: fz(64, &mut up),
            gf: fz(2 * D, &mut up),
            rl: fz(128, &mut up),
            ids: fz(TOP_K, &mut up),
            wts: fz(TOP_K, &mut up),
            act: fz(TOP_K * FF, &mut up),
            logits: fz(201024, &mut up),
        };
        let unembed = up.bytes(as_bytes(cpu.unembed));
        let norm = f32s(&model::bf_vec(cpu.norm), &mut up);
        let output_query = f32s(cpu.output_query, &mut up);
        eprintln!("GPU に {:.2} GB 載せた", up.total as f64 / 1e9);
        Self { cpu, _ctx: ctx, stream, _module: module, f, layers, slots, unembed, norm, output_query, b, pos: 0, graphs: RefCell::new(vec![]), capturing: Cell::new(false) }
    }

    fn launch(&self, name: &str, grid: u32, block: u32, smem: u32, args: &[u64]) {
        let mut vals = args.to_vec();
        let mut ptrs: Vec<*mut c_void> = vals.iter_mut().map(|v| v as *mut u64 as *mut c_void).collect();
        // SAFETY: 引数の数と型は gpu.cu の宣言に合わせてある。ポインタはどれも 64bit
        unsafe { result::launch_kernel(self.f[name].cu_function(), (grid, 1, 1), (block, 1, 1), smem, self.stream.cu_stream(), &mut ptrs) }
            .unwrap_or_else(|e| panic!("{name}: {e}"));
    }

    fn copy(&self, dst: P, src: P, floats: usize) {
        // SAFETY: どちらも確保済みの領域
        unsafe { result::memcpy_dtod_async(dst, src, floats * 4, self.stream.cu_stream()) }.unwrap();
    }

    fn htod(&self, dst: P, src: &[f32]) {
        // SAFETY: dst は src 以上の長さで確保してある。転送が終わるまで待つ
        unsafe { result::memcpy_htod_async(dst, src, self.stream.cu_stream()) }.unwrap();
        self.stream.synchronize().unwrap();
    }

    fn dtoh(&self, src: P, n: usize) -> Vec<f32> {
        let mut v = vec![0.0; n];
        // SAFETY: src は n 要素以上ある。転送が終わるまで待つ
        unsafe { result::memcpy_dtoh_async(&mut v, src, self.stream.cu_stream()) }.unwrap();
        self.stream.synchronize().unwrap();
        v
    }

    fn mv(&self, w: P, x: P, y: P, rows: usize, cols: usize) {
        k!(self, "mv_bf16", (rows * 32).div_ceil(256), 256, 0; w, x, y, rows as i32, cols as i32);
    }

    fn mix(&self, blocks: P, nb: usize, cur: P, queries: P, nq: usize, out: P) {
        k!(self, "mix", nq, 256, 0; blocks, nb as i32, cur, queries, out);
    }

    fn control(&self, cur: P, msg: P, c: &GCtrl) {
        let b = &self.b;
        k!(self, "norm2", 2, 256, 0; cur, msg, b.cat);
        self.mv(c.down, b.cat, b.g, 64, 2 * D);
        self.mv(c.gf, b.g, b.gf, 2 * D, 64);
        k!(self, "control", D.div_ceil(256), 256, 0; cur, msg, b.gf);
    }

    fn conv(&self, cur: P, w: P, dyn_: P, hist: P, out: P, ch: usize, cph: usize) {
        k!(self, "conv", ch.div_ceil(256), 256, 0; cur, w, dyn_, hist, out, ch as i32, cph as i32);
    }

    /// 1トークンを処理して、語彙の logits を返す。
    /// 最初のトークンでカーネル列を CUDA Graph に記録し、以降は再生する。トレース中は記録せず毎回カーネルを投げる。
    pub fn forward(&mut self, token: u32) -> Vec<f32> {
        assert!(self.pos < MAX_CTX, "文脈長の上限 {MAX_CTX} を超えた");
        let mut e = model::bf_row(self.cpu.embed, token as usize, D);
        model::rmsnorm(&mut e, Some(self.cpu.embed_norm));
        self.htod(self.b.x, &e);
        // SAFETY: b.pos は i32 1個ぶん確保してある。転送が終わるまで待つ
        unsafe { result::memcpy_htod_async(self.b.pos, &[self.pos as i32], self.stream.cu_stream()) }.unwrap();
        if self.graphs.borrow().is_empty() || model::tracing() {
            self.capturing.set(!model::tracing());
            self.begin();
            self.record();
            self.end();
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

    /// 20〜23層の MoE を CPU で計算する。b.h を読み b.y に書く
    fn cpu_moe(&self, il: usize) {
        let h = self.dtoh(self.b.h, D);
        let y = self.cpu.moe(&self.cpu.layers[il], &h);
        self.htod(self.b.y, &y);
    }

    fn record(&self) {
        let b = &self.b;
        let mut nb = 0;
        for il in 0..CORE.0 {
            self.layer(il, il, il, &mut nb, false);
        }
        self.copy(b.prelude, b.x, D);
        let prelude_nb = nb;
        for pass in 0..PASSES {
            if pass > 0 {
                // 前の周の、最初のコアブロックより後のブロックと出力
                assert_eq!(nb, prelude_nb + 2);
                self.copy(b.bank, b.blocks + ((prelude_nb + 1) * D * 4) as u64, D);
                self.copy(b.bank + (D * 4) as u64, b.x, D);
                self.copy(b.x, b.prelude, D);
                nb = prelude_nb;
            }
            for ip in CORE.0..CORE.1 {
                let slot = if pass == 0 { ip } else { N_LAYER + (pass - 1) * (CORE.1 - CORE.0) + ip - CORE.0 };
                if pass > 0 {
                    self.mix(b.bank, 1, b.bank + (D * 4) as u64, self.layers[ip].msg_query, 4, b.msgs);
                }
                self.layer(ip, ip, slot, &mut nb, pass > 0);
            }
        }
        for il in CORE.1..N_LAYER {
            self.layer(il, il, il, &mut nb, false);
        }
        self.mix(b.blocks, nb, b.x, self.output_query, 1, b.h);
        k!(self, "rmsnorm", 1, 256, 0; b.h, self.norm, b.h, D as i32, 0i32);
        self.mv(self.unembed, b.h, b.logits, 201024, D);
        k!(self, "softcap", VOCAB.div_ceil(256), 256, 0; b.logits, VOCAB as i32);
    }

    fn layer(&self, il: usize, ip: usize, slot: usize, nb: &mut usize, msgs: bool) {
        let (l, s, b) = (&self.layers[il], &self.slots[slot], &self.b);
        let fd = |i: usize| (i * D * 4) as u64; // D 要素 i 本ぶんのバイト数
        // attention の入力: SWA は1本、global は Q, K, V に1本ずつ
        if il > 0 {
            self.mix(b.blocks, *nb, b.x, l.attn_query, if l.global { 3 } else { 1 }, b.inp);
        } else {
            self.copy(b.inp, b.x, D);
        }
        let separate = l.global || msgs;
        if !l.global && msgs {
            self.copy(b.inp + fd(1), b.inp, D);
            self.copy(b.inp + fd(2), b.inp, D);
        }
        if msgs {
            for j in 0..3 {
                self.control(b.inp + fd(j), b.msgs + fd(j), &l.ctrl[j]);
            }
        }
        let n_in = if separate { 3 } else { 1 };
        k!(self, "rmsnorm", n_in, 256, 0; b.inp, l.norms, b.inp, D as i32, if l.global { D as i32 } else { 0i32 });
        let (xq, xk, xv) = if separate { (b.inp, b.inp + fd(1), b.inp + fd(2)) } else { (b.inp, b.inp, b.inp) };

        let nk = if l.global { 1024 + 32 } else { 1024 };
        self.mv(l.wq, xq, b.qx, QX, D);
        self.mv(l.wk, xk, b.kx, nk, D);
        self.mv(l.wv, xv, b.vx, nk, D);
        let (kd, vd) = if l.global { (b.kx + 4096, b.vx + 4096) } else { (0, 0) };
        self.conv(b.kx, l.k_conv, kd, s.hist[0], b.kp, 1024, 128);
        self.conv(b.vx, l.v_conv, vd, s.hist[1], b.vp, 1024, 128);
        let f4 = |n: usize| (n * 4) as u64;
        k!(self, "qkv_post", 64, 128, 0; b.qx, b.qx + f4(R_OFF), b.kp, b.vp, l.q_clip, l.rel_clip, b.pos, l.global as i32, s.kc, s.vc);
        let smem = (EXTENT + MAX_CTX) * 4;
        k!(self, "attn", 24, 128, smem; b.qx, b.qx + f4(R_OFF), b.qx + f4(SINK_OFF), s.kc, s.vc, l.rel_proj, b.qx + f4(GATE_OFF), b.o,
            b.pos, (!l.global) as i32);
        self.mv(l.wo, b.o, b.a, D, 3072);
        self.conv(b.a, l.attn_conv, 0, s.hist[2], b.a, D, D);
        // ブロックの先頭層は残差を引き継がず、入力をブロック列に積む
        if ip % BLOCK == 0 {
            self.copy(b.blocks + fd(*nb), b.x, D);
            *nb += 1;
            self.copy(b.x, b.a, D);
        } else {
            k!(self, "add", D.div_ceil(256), 256, 0; b.x, b.a, b.x, D as i32);
        }

        self.mix(b.blocks, *nb, b.x, l.mlp_query, 1, b.h);
        if msgs {
            self.control(b.h, b.msgs + fd(3), &l.ctrl[3]);
        }
        k!(self, "rmsnorm", 1, 256, 0; b.h, l.mlp_norm, b.h, D as i32, 0i32);
        match l.experts {
            Some([w13, s13, a13, w2, s2, a2]) => {
                self.mv(l.router, b.h, b.rl, 128, D);
                k!(self, "topk", 1, 128, 0; b.rl, l.router_bias, b.ids, b.wts);
                k!(self, "moe_up_nvfp4", (TOP_K * FF * 32).div_ceil(256), 256, 0; w13, s13, a13, b.ids, b.h, b.act);
                k!(self, "moe_down_nvfp4", (D * 32).div_ceil(256), 256, 0; w2, s2, a2, b.ids, b.wts, b.act, b.y);
            }
            None => {
                self.end();
                self.cpu_moe(il);
                self.begin();
            }
        }
        self.conv(b.y, l.mlp_conv, 0, s.hist[3], b.y, D, D);
        k!(self, "add", D.div_ceil(256), 256, 0; b.x, b.y, b.x, D as i32);
        if model::tracing() {
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
    fn new(_ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>) -> Self {
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
