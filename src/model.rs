//! Inkling 2 Mini のテキスト部の順伝播。1トークンずつ処理する。
//! 仕様は llama.cpp の src/models/inkling2.cpp と inkling2-loop.cpp に合わせている。

use crate::kernels::{dequant_mxfp8, dot_2x4, dot_bf16, dot_mxfp8, dot_nvfp4};
use crate::st::SafeTensors;

pub const D: usize = 2304;
pub(crate) const N_LAYER: usize = 24;
pub(crate) const N_HEAD: usize = 24;
pub(crate) const N_KV: usize = 8;
pub(crate) const HD: usize = 128;
pub(crate) const N_REL: usize = 16;
pub(crate) const EXTENT: usize = 1024; // 相対位置表の長さ。SWA の窓幅も同じ 1024
pub(crate) const N_EXP: usize = 128;
pub(crate) const TOP_K: usize = 8;
pub(crate) const FF: usize = 1152;
pub(crate) const RANK: usize = 64;
pub(crate) const CORE: (usize, usize) = (6, 18);
pub(crate) const PASSES: usize = 3;
pub(crate) const BLOCK: usize = 6; // attention residual の1ブロックの層数
pub(crate) const N_SLOT: usize = N_LAYER + (PASSES - 1) * (CORE.1 - CORE.0);
pub(crate) const VOCAB: usize = 200058;
pub(crate) const EPS: f32 = 1e-6;
pub(crate) const QK_CLIP: f32 = 22.627416997969522;
pub(crate) const REL_CLIP: f32 = 8.0;
pub(crate) const ROUTE_SCALE: f32 = 8.0;
pub(crate) const SITU_BETA: f32 = 4.0;
pub(crate) const SITU_LINEAR: f32 = 25.0;
pub(crate) const SOFTCAP: f32 = 30.0;
pub(crate) const MXFP8_FROM: usize = 20; // 20〜23層の expert は MXFP8
pub(crate) const GLOBAL_LAYERS: [usize; 4] = [5, 11, 17, 23];

pub(crate) type W = &'static [u16]; // BF16 の行優先行列 [out, in]

pub(crate) struct Experts {
    pub(crate) mx: bool, // true なら MXFP8、false なら NVFP4
    pub(crate) w13: &'static [u8],
    pub(crate) s13: &'static [u8],
    pub(crate) a13: &'static [f32],
    pub(crate) w2: &'static [u8],
    pub(crate) s2: &'static [u8],
    pub(crate) a2: &'static [f32],
}

pub(crate) struct Ctrl {
    pub(crate) down: W,
    pub(crate) gain: W,
    pub(crate) forget: W,
}

pub(crate) struct Layer {
    pub(crate) global: bool,
    pub(crate) attn_query: Option<&'static [f32]>, // 1本 (SWA) か 3本 (global)
    pub(crate) mlp_query: &'static [f32],
    pub(crate) norms: [W; 3], // SWA は3つとも attn_norm
    pub(crate) wq: W,
    pub(crate) wk: W,
    pub(crate) wv: W,
    pub(crate) wo: W,
    pub(crate) wr: W,
    pub(crate) gate_attn: W,
    pub(crate) sink: W,
    pub(crate) q_clip: W,
    pub(crate) rel_clip: W,
    pub(crate) rel_proj: W, // [N_REL, EXTENT]
    pub(crate) k_conv: Vec<[f32; 4]>, // [ch][k]、k=3 が現在のトークン
    pub(crate) v_conv: Vec<[f32; 4]>,
    pub(crate) k_dyn: Option<W>,
    pub(crate) v_dyn: Option<W>,
    pub(crate) attn_conv: Vec<[f32; 4]>,
    pub(crate) mlp_conv: Vec<[f32; 4]>,
    pub(crate) mlp_norm: W,
    pub(crate) router: W,
    pub(crate) router_bias: &'static [f32],
    pub(crate) experts: Experts,
    pub(crate) ctrl: Option<[Ctrl; 4]>, // q, k, v, mlp
    pub(crate) msg_query: Option<Vec<f32>>, // [4, D]
}

pub struct Model {
    pub(crate) embed: W,
    pub(crate) embed_norm: W,
    pub(crate) norm: W,
    pub(crate) unembed: W,
    pub(crate) output_query: &'static [f32],
    pub(crate) layers: Vec<Layer>,
}

#[derive(Default, Clone)]
struct Slot {
    k: Vec<f32>, // [pos, N_KV*HD]
    v: Vec<f32>,
    hist: [Vec<[f32; 3]>; 4], // k, v, attn 出力, mlp 出力 の直前3トークン分の入力
}

pub struct State {
    pos: usize,
    slots: Vec<Slot>,
}

impl State {
    pub fn new() -> Self {
        let mut s = Slot::default();
        s.hist = [vec![[0.0; 3]; N_KV * HD], vec![[0.0; 3]; N_KV * HD], vec![[0.0; 3]; D], vec![[0.0; 3]; D]];
        Self { pos: 0, slots: vec![s; N_SLOT] }
    }
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) fn cast<T>(b: &'static [u8]) -> &'static [T] {
    // SAFETY: u16/f32 はどのビット列も有効。整列は下で検査する
    let (pre, mid, post) = unsafe { b.align_to::<T>() };
    assert!(pre.is_empty() && post.is_empty(), "テンソルが整列していない");
    mid
}

impl Model {
    pub fn load(st: SafeTensors) -> Self {
        let st: &'static SafeTensors = Box::leak(Box::new(st));
        let get = |name: &str, dtype: &str, shape: &[usize]| -> &'static [u8] {
            let t = st.get(name).unwrap_or_else(|| panic!("{name} がない"));
            assert_eq!((t.dtype.as_str(), t.shape.as_slice()), (dtype, shape), "{name}");
            t.data
        };
        let bf = |name: &str, shape: &[usize]| -> W { cast(get(name, "BF16", shape)) };
        let f32s = |name: &str, shape: &[usize]| -> &'static [f32] { cast(get(name, "F32", shape)) };
        let bf_f32 = |name: &str, shape: &[usize]| -> Vec<f32> { bf(name, shape).iter().map(|&x| bf16(x)).collect() };
        let conv = |name: &str, ch: usize| -> Vec<[f32; 4]> {
            bf_f32(name, &[ch, 1, 4]).chunks(4).map(|c| c.try_into().unwrap()).collect()
        };
        // static_w は [lag, ch] で lag=0 が現在のトークン
        let static_conv = |name: &str, ch: usize| -> Vec<[f32; 4]> {
            let w = bf_f32(name, &[4, ch]);
            (0..ch).map(|c| [w[3 * ch + c], w[2 * ch + c], w[ch + c], w[c]]).collect()
        };
        let q = f32s("model.llm.middle_cycle_cross_pass_queries", &[24, D]);
        let kvq = f32s("model.llm.middle_cycle_cross_pass_kv_queries", &[24, D]);

        let layers = (0..N_LAYER)
            .map(|i| {
                let p = format!("model.llm.layers.{i}.");
                let global = GLOBAL_LAYERS.contains(&i);
                let n = |s: &str| format!("{p}{s}");
                let in_core = (CORE.0..CORE.1).contains(&i);
                let ctrl = |c: &str| Ctrl {
                    down: bf(&n(&format!("controllers.{c}.down_weight")), &[RANK, 2 * D]),
                    gain: bf(&n(&format!("controllers.{c}.gain_weight")), &[D, RANK]),
                    forget: bf(&n(&format!("controllers.{c}.forget_weight")), &[D, RANK]),
                };
                let norms = if global {
                    [bf(&n("q_input_norm.weight"), &[D]), bf(&n("k_input_norm.weight"), &[D]), bf(&n("v_input_norm.weight"), &[D])]
                } else {
                    let w = bf(&n("attn_norm.weight"), &[D]);
                    [w, w, w]
                };
                let (k_conv, v_conv, k_dyn, v_dyn) = if global {
                    (
                        static_conv(&n("attn.k_sconv.static_w"), N_KV * HD),
                        static_conv(&n("attn.v_sconv.static_w"), N_KV * HD),
                        Some(bf(&n("attn.k_sconv.weight_proj"), &[4 * N_KV, D])),
                        Some(bf(&n("attn.v_sconv.weight_proj"), &[4 * N_KV, D])),
                    )
                } else {
                    (conv(&n("attn.k_sconv.weight"), N_KV * HD), conv(&n("attn.v_sconv.weight"), N_KV * HD), None, None)
                };
                let e = |w: &str| format!("{p}mlp.experts.{w}");
                Layer {
                    global,
                    attn_query: (i > 0).then(|| {
                        let name = format!("model.llm.attn_res.attn_queries.{}", i - 1);
                        if global { f32s(&name, &[3, D]) } else { f32s(&name, &[D]) }
                    }),
                    mlp_query: f32s(&format!("model.llm.attn_res.mlp_queries.{i}"), &[D]),
                    norms,
                    wq: bf(&n("attn.wq_du.weight"), &[N_HEAD * HD, D]),
                    wk: bf(&n("attn.wk_dv.weight"), &[N_KV * HD, D]),
                    wv: bf(&n("attn.wv_dv.weight"), &[N_KV * HD, D]),
                    wo: bf(&n("attn.wo_ud.weight"), &[D, N_HEAD * HD]),
                    wr: bf(&n("attn.wr_du.weight"), &[N_REL * N_HEAD, D]),
                    gate_attn: bf(&n("attn.gate_attn.weight"), &[N_HEAD * HD, D]),
                    sink: bf(&n("attn.attn_sink_proj.weight"), &[N_HEAD, D]),
                    q_clip: bf(&n("attn.q_l2_clip.scale"), &[N_HEAD, HD]),
                    rel_clip: bf(&n("attn.rel_l2_clip.scale"), &[N_HEAD, N_REL]),
                    rel_proj: bf(&n("attn.rel_logits_proj.proj"), &[N_REL, EXTENT]),
                    k_conv,
                    v_conv,
                    k_dyn,
                    v_dyn,
                    attn_conv: conv(&n("attn_sconv.weight"), D),
                    mlp_conv: conv(&n("mlp_sconv.weight"), D),
                    mlp_norm: bf(&n("mlp_norm.weight"), &[D]),
                    router: bf(&n("mlp.gate.weight"), &[N_EXP, D]),
                    router_bias: f32s(&n("mlp.gate.bias"), &[N_EXP]),
                    experts: if i >= MXFP8_FROM {
                        Experts {
                            mx: true,
                            w13: get(&e("w13_weight"), "F8_E4M3", &[N_EXP, 2 * FF, D]),
                            s13: get(&e("w13_weight_scale"), "U8", &[N_EXP, 2 * FF, D / 32]),
                            a13: &[],
                            w2: get(&e("w2_weight"), "F8_E4M3", &[N_EXP, D, FF]),
                            s2: get(&e("w2_weight_scale"), "U8", &[N_EXP, D, FF / 32]),
                            a2: &[],
                        }
                    } else {
                        Experts {
                            mx: false,
                            w13: get(&e("w13_weight"), "U8", &[N_EXP, 2 * FF, D / 2]),
                            s13: get(&e("w13_weight_scale"), "F8_E4M3", &[N_EXP, 2 * FF, D / 16]),
                            a13: f32s(&e("w13_weight_amax"), &[N_EXP]),
                            w2: get(&e("w2_weight"), "U8", &[N_EXP, D, FF / 2]),
                            s2: get(&e("w2_weight_scale"), "F8_E4M3", &[N_EXP, D, FF / 16]),
                            a2: f32s(&e("w2_weight_amax"), &[N_EXP]),
                        }
                    },
                    ctrl: in_core.then(|| [ctrl("attn"), ctrl("k"), ctrl("v"), ctrl("mlp")]),
                    msg_query: in_core.then(|| {
                        let j = i - CORE.0;
                        [&q[2 * j * D..], &kvq[2 * j * D..], &kvq[(2 * j + 1) * D..], &q[(2 * j + 1) * D..]]
                            .iter()
                            .flat_map(|r| r[..D].iter().copied())
                            .collect()
                    }),
                }
            })
            .collect();
        // moe() は分母を 2688 と決め打ちしている
        for i in 0..MXFP8_FROM {
            for w in ["w13", "w2"] {
                let d = f32s(&format!("model.llm.layers.{i}.mlp.experts.{w}_weight_outer_scale_denominator"), &[1]);
                assert_eq!(d[0], 2688.0);
            }
        }
        Self {
            embed: bf("model.llm.embed.weight", &[201024, D]),
            embed_norm: bf("model.llm.embed_norm.weight", &[D]),
            norm: bf("model.llm.norm.weight", &[D]),
            unembed: bf("model.llm.unembed.weight", &[201024, D]),
            output_query: f32s("model.llm.attn_res.output_query", &[D]),
            layers,
        }
    }

    /// 1トークンを処理して、語彙の logits を返す。
    pub fn forward(&self, st: &mut State, token: u32) -> Vec<f32> {
        let pos = st.pos;
        let mut x = bf_row(self.embed, token as usize, D);
        rmsnorm(&mut x, Some(self.embed_norm));
        let mut blocks: Vec<Vec<f32>> = vec![];
        for il in 0..CORE.0 {
            x = self.layer(il, il, &mut st.slots[il], pos, x, &mut blocks, None);
            trace(il, &x);
        }
        let prelude = x.clone();
        let prelude_blocks = blocks.clone();
        for pass in 0..PASSES {
            let mut bank = vec![];
            if pass > 0 {
                bank = blocks[prelude_blocks.len() + 1..].to_vec();
                bank.push(std::mem::replace(&mut x, prelude.clone()));
                blocks = prelude_blocks.clone();
            }
            for ip in CORE.0..CORE.1 {
                let slot = if pass == 0 { ip } else { N_LAYER + (pass - 1) * (CORE.1 - CORE.0) + ip - CORE.0 };
                let msgs = (pass > 0).then(|| mix(&bank, self.layers[ip].msg_query.as_ref().unwrap()));
                x = self.layer(ip, ip, &mut st.slots[slot], pos, x, &mut blocks, msgs.as_deref());
                trace(slot, &x);
            }
        }
        for il in CORE.1..N_LAYER {
            x = self.layer(il, il, &mut st.slots[il], pos, x, &mut blocks, None);
            trace(il, &x);
        }
        blocks.push(x);
        let mut h = mix(&blocks, self.output_query).pop().unwrap();
        rmsnorm(&mut h, Some(self.norm));
        let mut logits = vec![0.0; VOCAB];
        matvec(self.unembed, &h, &mut logits);
        for l in &mut logits {
            *l = SOFTCAP * (*l / SOFTCAP).tanh();
        }
        st.pos += 1;
        logits
    }

    #[allow(clippy::too_many_arguments)]
    fn layer(&self, il: usize, ip: usize, slot: &mut Slot, pos: usize, cur: Vec<f32>, blocks: &mut Vec<Vec<f32>>, msgs: Option<&[Vec<f32>]>) -> Vec<f32> {
        let l = &self.layers[il];
        let mut inputs = [cur.clone(), cur.clone(), cur.clone()];
        if let Some(q) = l.attn_query {
            let mut states = blocks.clone();
            states.push(cur.clone());
            let mixed = mix(&states, q);
            for (j, inp) in inputs.iter_mut().enumerate() {
                *inp = mixed[if l.global { j } else { 0 }].clone();
            }
        }
        if let Some(m) = msgs {
            let c = l.ctrl.as_ref().unwrap();
            for j in 0..3 {
                inputs[j] = control(&inputs[j], &m[j], &c[j]);
            }
        }
        for (inp, w) in inputs.iter_mut().zip(l.norms) {
            rmsnorm(inp, Some(w));
        }
        let [xq, xk, xv] = &inputs;

        let mut q = mv(l.wq, xq);
        let mut r = mv(l.wr, xq);
        let gate = mv(l.gate_attn, xq);
        let sinks = mv(l.sink, xq);
        let k = mv(l.wk, xk);
        let v = mv(l.wv, xv);
        let mut k = convolve(&k, &l.k_conv, l.k_dyn.map(|w| mv(w, xk)).as_deref(), &mut slot.hist[0]);
        let mut v = convolve(&v, &l.v_conv, l.v_dyn.map(|w| mv(w, xv)).as_deref(), &mut slot.hist[1]);
        let qc = bf_vec(l.q_clip);
        let rc = bf_vec(l.rel_clip);
        for h in 0..N_HEAD {
            clip(&mut q[h * HD..][..HD], Some(&qc[h * HD..][..HD]), QK_CLIP);
            clip(&mut r[h * N_REL..][..N_REL], Some(&rc[h * N_REL..][..N_REL]), REL_CLIP);
        }
        for h in 0..N_KV {
            clip(&mut k[h * HD..][..HD], None, QK_CLIP);
            rmsnorm(&mut v[h * HD..][..HD], None);
        }
        if l.global {
            let tau = 1.0 + 0.1 * ((pos as f32 + 1.0) / 8192.0).max(1.0).ln();
            q.iter_mut().chain(r.iter_mut()).for_each(|x| *x *= tau);
        }
        slot.k.extend_from_slice(&k);
        slot.v.extend_from_slice(&v);

        let n = pos + 1;
        let first = if l.global { 0 } else { n.saturating_sub(EXTENT) };
        let rel_proj = bf_vec(l.rel_proj);
        let scale = 1.0 / (HD as f32).sqrt();
        let mut o = vec![0.0; N_HEAD * HD];
        crate::pool::chunks_mut(&mut o, HD, |h, oh| {
            let qh = &q[h * HD..][..HD];
            let rh = &r[h * N_REL..][..N_REL];
            let kvh = h / (N_HEAD / N_KV);
            // 距離 d の相対位置バイアス。d >= EXTENT は 0
            let rel: Vec<f32> = (0..EXTENT).map(|d| (0..N_REL).map(|i| rh[i] * rel_proj[i * EXTENT + d]).sum()).collect();
            let s: Vec<f32> = (first..n)
                .map(|j| {
                    let kj = &slot.k[(j * N_KV + kvh) * HD..][..HD];
                    let d = pos - j;
                    dot(qh, kj) * scale + if d < EXTENT { rel[d] } else { 0.0 }
                })
                .collect();
            // sink のロジットは sinks[h]、値はゼロベクトル
            let m = s.iter().copied().fold(sinks[h], f32::max);
            let mut denom = (sinks[h] - m).exp();
            for (j, &sj) in (first..n).zip(&s) {
                let p = (sj - m).exp();
                denom += p;
                let vj = &slot.v[(j * N_KV + kvh) * HD..][..HD];
                oh.iter_mut().zip(vj).for_each(|(o, v)| *o += p * v);
            }
            oh.iter_mut().for_each(|o| *o /= denom);
        });
        o.iter_mut().zip(&gate).for_each(|(o, g)| *o *= sigmoid(*g));
        let a = convolve(&mv(l.wo, &o), &l.attn_conv, None, &mut slot.hist[2]);

        // ブロックの先頭層は残差を引き継がず、入力をブロック列に積む
        let cur = if ip % BLOCK == 0 {
            blocks.push(cur);
            a
        } else {
            add(&cur, &a)
        };

        let mut states = blocks.clone();
        states.push(cur.clone());
        let mut h = mix(&states, l.mlp_query).pop().unwrap();
        if let Some(m) = msgs {
            h = control(&h, &m[3], &l.ctrl.as_ref().unwrap()[3]);
        }
        rmsnorm(&mut h, Some(l.mlp_norm));
        let m = convolve(&self.moe(l, &h), &l.mlp_conv, None, &mut slot.hist[3]);
        add(&cur, &m)
    }

    /// ルーター: 上位8個の expert と、正規化して route_scale を掛けた重み
    pub(crate) fn route(&self, l: &Layer, x: &[f32]) -> ([usize; TOP_K], [f32; TOP_K]) {
        let p: Vec<f32> = mv(l.router, x).iter().map(|&z| sigmoid(z)).collect();
        let mut idx: Vec<usize> = (0..N_EXP).collect();
        idx.sort_by(|&a, &b| (p[b] + l.router_bias[b]).total_cmp(&(p[a] + l.router_bias[a])));
        let sel: [usize; TOP_K] = idx[..TOP_K].try_into().unwrap();
        let sum: f32 = sel.iter().map(|&e| p[e]).sum::<f32>() + 1e-20;
        (sel, sel.map(|e| p[e] / sum * ROUTE_SCALE))
    }

    pub(crate) fn moe(&self, l: &Layer, x: &[f32]) -> Vec<f32> {
        let (sel, wts) = self.route(l, x);
        self.moe_some(l, x, &sel, &wts)
    }

    /// 1トークンの MoE のうち、expert sel (重み wts、route_scale 込み) の分の和
    pub(crate) fn moe_some(&self, l: &Layer, x: &[f32], sel: &[usize], wts: &[f32]) -> Vec<f32> {
        let n = sel.len();
        let e = &l.experts;
        // 選ばれた expert の行をまとめて分割し、全スレッドに配る
        let mut gu = vec![0.0; n * 2 * FF];
        crate::pool::chunks_mut(&mut gu, 64, |c, o| {
            for (i, o) in o.iter_mut().enumerate() {
                let r = c * 64 + i;
                let ex = sel[r / (2 * FF)];
                *o = e.dot(e.w13, e.s13, ex * 2 * FF + r % (2 * FF), x) * e.g13(ex);
            }
        });
        let act = situ_pairs(&gu);
        let g2: Vec<f32> = (0..n).map(|k| wts[k] * e.g2(sel[k])).collect();
        let mut y = vec![0.0; D];
        crate::pool::chunks_mut(&mut y, 64, |c, o| {
            for (i, o) in o.iter_mut().enumerate() {
                let row = c * 64 + i;
                *o = (0..n).map(|k| e.dot(e.w2, e.s2, sel[k] * D + row, &act[k * FF..][..FF]) * g2[k]).sum();
            }
        });
        y
    }

    /// T トークンぶんの MoE のうち、mine(expert) が真の expert の分。expert ごとに割り当てられたトークンをまとめ、
    /// 重みの行を読むあいだに全トークンへ掛ける。割り当て a = トークン * 8 + k の expert は ids[a]、重みは wts[a]
    /// (route_scale 込み)。ルーティングは GPU が求めたものを受け取り、GPU と CPU で expert の分担が食い違わないようにする。
    pub(crate) fn moe_batch(&self, l: &Layer, xs: &[f32], ids: &[i32], wts: &[f32], mine: impl Fn(usize) -> bool) -> Vec<f32> {
        let t = xs.len() / D;
        let e = &l.experts;
        let mut lists = vec![vec![]; N_EXP];
        for (a, &ex) in ids.iter().enumerate() {
            if mine(ex as usize) {
                lists[ex as usize].push(a);
            }
        }
        let used: Vec<usize> = (0..N_EXP).filter(|&ex| !lists[ex].is_empty()).collect();
        let n_assign = t * TOP_K;

        let mut gu = vec![0.0; n_assign * 2 * FF];
        let out = Shared(gu.as_mut_ptr());
        let chunks = 2 * FF / 64;
        crate::pool::run(used.len() * chunks, &|task| {
            let ex = used[task / chunks];
            let r0 = (task % chunks) * 64;
            block_gemm(e, e.w13, e.s13, ex * 2 * FF + r0, D, &lists[ex], |a| &xs[a / TOP_K * D..][..D], |a, r, v| {
                // SAFETY: (expert, 行) ごとに書く場所は重ならない
                unsafe { *out.at(a * 2 * FF + r0 + r) = v * e.g13(ex) };
            });
        });
        let mut act = vec![0.0; n_assign * FF];
        crate::pool::chunks_mut(&mut act, FF, |a, o| o.copy_from_slice(&situ_pairs(&gu[a * 2 * FF..][..2 * FF])));

        let mut yd = vec![0.0; n_assign * D];
        let out = Shared(yd.as_mut_ptr());
        let chunks = D / 64;
        crate::pool::run(used.len() * chunks, &|task| {
            let ex = used[task / chunks];
            let r0 = (task % chunks) * 64;
            block_gemm(e, e.w2, e.s2, ex * D + r0, FF, &lists[ex], |a| &act[a * FF..][..FF], |a, r, v| {
                // SAFETY: 同上
                unsafe { *out.at(a * D + r0 + r) = v * e.g2(ex) * wts[a] };
            });
        });
        let mut y = vec![0.0; t * D];
        crate::pool::chunks_mut(&mut y, D, |tok, o| {
            for k in 0..TOP_K {
                o.iter_mut().zip(&yd[(tok * TOP_K + k) * D..][..D]).for_each(|(o, v)| *o += v);
            }
        });
        y
    }
}

/// expert の重みの 64 行 (row0 から) と、list の各割り当ての入力 x(a) (長さ k) の積を out(a, 行, 値) に渡す。
/// 64 行を f32 に展開し、列を KB ずつに区切って 2行 x 4トークンの組で積和する。
/// 4トークン分の x の区間 (4 x KB x 4 バイト) が L1 に残り、重みの行は L2 から読む。
fn block_gemm<'a>(e: &Experts, w: &[u8], s: &[u8], row0: usize, k: usize, list: &[usize], x: impl Fn(usize) -> &'a [f32], mut out: impl FnMut(usize, usize, f32)) {
    const ROWS: usize = 64;
    const KB: usize = 1152;
    // 割り当てが少ない (検証の小さいバッチ) と、行を f32 に展開する手間のほうが重いので、量子化のまま内積をとる
    if list.len() < 8 {
        for r in 0..ROWS {
            for &a in list {
                out(a, r, e.dot(w, s, row0 + r, x(a)));
            }
        }
        return;
    }
    let mut wb = vec![0.0; ROWS * k];
    for (r, o) in wb.chunks_mut(k).enumerate() {
        e.dequant(w, s, row0 + r, o);
    }
    // 4 の倍数になるよう最後の割り当てを繰り返す (余りの結果は捨てる)
    let mut l4 = list.to_vec();
    while l4.len() % 4 != 0 {
        l4.push(*list.last().unwrap());
    }
    let n4 = l4.len();
    let mut acc = vec![0.0; ROWS * n4];
    for kb in (0..k).step_by(KB) {
        let kl = KB.min(k - kb);
        for (g, tg) in l4.chunks(4).enumerate() {
            let xs = [0, 1, 2, 3].map(|j| &x(tg[j])[kb..kb + kl]);
            for r in (0..ROWS).step_by(2) {
                let v = dot_2x4([&wb[r * k + kb..][..kl], &wb[(r + 1) * k + kb..][..kl]], xs);
                for j in 0..4 {
                    acc[r * n4 + g * 4 + j] += v[0][j];
                    acc[(r + 1) * n4 + g * 4 + j] += v[1][j];
                }
            }
        }
    }
    for r in 0..ROWS {
        for (j, &a) in list.iter().enumerate() {
            out(a, r, acc[r * n4 + j]);
        }
    }
}

/// スレッド間で、重ならない場所に書くためのポインタ
struct Shared(*mut f32);
unsafe impl Sync for Shared {}
impl Shared {
    fn at(&self, i: usize) -> *mut f32 {
        // SAFETY: 呼び出し側が範囲内を指定する
        unsafe { self.0.add(i) }
    }
}

/// gate と up が交互に並ぶ列に SITU をかける
fn situ_pairs(gu: &[f32]) -> Vec<f32> {
    gu.chunks(2).map(|c| SITU_BETA * (c[0] / SITU_BETA).tanh() * sigmoid(c[0]) * SITU_LINEAR * (c[1] / SITU_LINEAR).tanh()).collect()
}

fn bf16(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

pub(crate) fn bf_vec(w: W) -> Vec<f32> {
    w.iter().map(|&x| bf16(x)).collect()
}

pub(crate) fn bf_row(w: W, row: usize, n: usize) -> Vec<f32> {
    bf_vec(&w[row * n..][..n])
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

fn matvec(w: W, x: &[f32], out: &mut [f32]) {
    let n = x.len();
    crate::pool::chunks_mut(out, 64, |c, o| {
        for (i, o) in o.iter_mut().enumerate() {
            *o = dot_bf16(&w[(c * 64 + i) * n..][..n], x);
        }
    });
}

fn mv(w: W, x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; w.len() / x.len()];
    matvec(w, x, &mut out);
    out
}

impl Experts {
    /// gate/up と down の大域スケール。MXFP8 は 1
    fn g13(&self, ex: usize) -> f32 {
        if self.mx { 1.0 } else { self.a13[ex] / 2688.0 }
    }

    fn g2(&self, ex: usize) -> f32 {
        if self.mx { 1.0 } else { self.a2[ex] / 2688.0 }
    }

    /// 行 row を f32 に展開する。NVFP4 は amax の大域スケールを掛ける前の値
    fn dequant(&self, w: &[u8], s: &[u8], row: usize, out: &mut [f32]) {
        let n = out.len();
        if self.mx {
            dequant_mxfp8(&w[row * n..][..n], &s[row * n / 32..][..n / 32], out)
        } else {
            crate::nvfp4::dequant_row(&w[row * n / 2..][..n / 2], &s[row * n / 16..][..n / 16], 1.0, out)
        }
    }

    /// 行 row と x の内積。NVFP4 は amax の大域スケールを掛ける前の値を返す。
    fn dot(&self, w: &[u8], s: &[u8], row: usize, x: &[f32]) -> f32 {
        let n = x.len();
        if self.mx {
            dot_mxfp8(&w[row * n..][..n], &s[row * n / 32..][..n / 32], x)
        } else {
            dot_nvfp4(&w[row * n / 2..][..n / 2], &s[row * n / 16..][..n / 16], x)
        }
    }
}

pub(crate) fn rmsnorm(x: &mut [f32], w: Option<W>) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let k = 1.0 / (ms + EPS).sqrt();
    match w {
        Some(w) => x.iter_mut().zip(w).for_each(|(x, &w)| *x *= k * bf16(w)),
        None => x.iter_mut().for_each(|x| *x *= k),
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(a, b)| a + b).collect()
}

/// 行の L2 ノルムを threshold 以下に縮める。scale があれば先に掛ける。
fn clip(x: &mut [f32], scale: Option<&[f32]>, threshold: f32) {
    if let Some(s) = scale {
        x.iter_mut().zip(s).for_each(|(x, s)| *x *= s);
    }
    let norm = x.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-6;
    let f = (threshold / norm).clamp(0.0, 1.0);
    x.iter_mut().for_each(|v| *v *= f);
}

/// 深さ方向の注意。各状態を RMS 正規化し、query ごとに softmax で重み付けした和を返す。
fn mix(states: &[Vec<f32>], queries: &[f32]) -> Vec<Vec<f32>> {
    let normed: Vec<Vec<f32>> = states
        .iter()
        .map(|s| {
            let mut s = s.clone();
            rmsnorm(&mut s, None);
            s
        })
        .collect();
    let scale = 1.0 / (D as f32).sqrt();
    queries
        .chunks(D)
        .map(|q| {
            let sc: Vec<f32> = normed.iter().map(|s| dot(s, q) * scale).collect();
            let m = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = sc.iter().map(|s| (s - m).exp()).collect();
            let z: f32 = e.iter().sum();
            let mut out = vec![0.0; D];
            for (s, p) in normed.iter().zip(&e) {
                out.iter_mut().zip(s).for_each(|(o, s)| *o += p / z * s);
            }
            out
        })
        .collect()
}

/// 因果的な短い畳み込みと残差。dynamic はヘッドごとの入力依存タップ [head][lag]。
fn convolve(cur: &[f32], w: &[[f32; 4]], dynamic: Option<&[f32]>, hist: &mut [[f32; 3]]) -> Vec<f32> {
    let heads = dynamic.map_or(1, |d| d.len() / 4);
    let ch_per_head = cur.len() / heads;
    let k = if dynamic.is_some() { 9.0f32 } else { 5.0 };
    let out = cur
        .iter()
        .enumerate()
        .map(|(c, &x)| {
            let h = &hist[c];
            let win = [h[0], h[1], h[2], x]; // t-3, t-2, t-1, t
            let mut o: f32 = (0..4).map(|i| w[c][i] * win[i]).sum();
            if let Some(d) = dynamic {
                let head = c / ch_per_head;
                o += (0..4).map(|lag| d[head * 4 + lag] * win[3 - lag]).sum::<f32>();
            }
            (x + o) / k.sqrt()
        })
        .collect();
    for (h, &x) in hist.iter_mut().zip(cur) {
        *h = [h[1], h[2], x];
    }
    out
}

/// ループの2周目以降で、前の周からのメッセージを入力に混ぜる。
fn control(cur: &[f32], msg: &[f32], c: &Ctrl) -> Vec<f32> {
    let mut a = cur.to_vec();
    let mut b = msg.to_vec();
    rmsnorm(&mut a, None);
    rmsnorm(&mut b, None);
    a.extend_from_slice(&b);
    let g = mv(c.down, &a);
    let gain = mv(c.gain, &g);
    let forget = mv(c.forget, &g);
    (0..D).map(|i| cur[i] - forget[i].tanh() * cur[i] + gain[i].tanh() * msg[i]).collect()
}

static TRACE: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var_os("OCTO_TRACE").is_some());

/// OCTO_TRACE を設定すると、各キャッシュスロットの層出力を llama.cpp の l_out-N と比べられる形で出す。
pub(crate) fn trace(slot: usize, x: &[f32]) {
    if *TRACE {
        eprintln!("l_out-{slot} [{:.4} {:.4} {:.4}] sum {:.6}", x[0], x[1], x[2], x.iter().sum::<f32>());
    }
}

pub(crate) fn tracing() -> bool {
    *TRACE
}
