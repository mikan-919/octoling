//! 下書きモデル DFlash2 (incoai/Inkling-2-Mini-DFlash2 の model.safetensors を dflash2.safetensors として置く)。
//! 計算は z-lab/dflash の dflash/model.py (commit 07ebd93) と、SGLang の DFlash2 (srt/models/dflash.py) と、
//! 同じリポジトリの sglang_inkling_dflash2.py (Inkling 用の差分) に合わせた:
//!   文脈: 本体の4つの層出力 (taps。DSpark と同じ target_layer_ids 5, 23, 41, 48) を並べて fc で射影し、hidden_norm で正規化したものを、
//!     各層の k_proj, v_proj (ブロックと同じ重み)、k_norm、RoPE (base 1e4) に通して文脈の K/V にする
//!   ブロック: 行 0 = 本体が選んだがまだ処理していないトークン (位置 P)、行 1..7 = マスクトークン (200056)。どちらも本体の embed と embed_norm を通す
//!   5層。各層は注意と MLP のそれぞれの前後を2タップの動的畳み込みで包む (dconv)。注意はブロックの中はマスクなし、文脈は位置の差が 2048 未満のキー
//!   出力: 行 i (1..7) が位置 P+i のトークン。正規化した隠れ状態を本体の出力層に通した logits の上位 16 個を候補にし、
//!     前のトークンとの組の点数を足して、1つずつ選ぶ (sel_walk)。softcap 30 は点数を足した後にかける (selector_softcap_mode = after_correction)
//! 下書きは貪欲法で選ぶ。本体は検証で各行を自分の分布から選び、下書きと一致した所までを受け入れるので、出力の分布は変わらない。

use super::draft::{DR_SPLIT, Fp8, fp8, tau};
use super::*;
use crate::st::SafeTensors;
use std::path::Path;

const B: usize = 8; // ブロックの行数
const L: usize = 5;
const G: usize = 4; // KV ヘッドあたりの q のヘッド数
const NQ: usize = 8 * G;
const HD: usize = 128;
const QKV: usize = NQ * HD + 2048;
const FF: usize = 6912;
const KP: usize = 4 * D / 16; // kernel_projection の出力 [2][2][D / 16]
const WIN: usize = 2048; // sliding window。文脈の K/V はこの行数のリングバッファに置く
const THETA: f32 = 1e4;
const MASK: usize = 200056;
const K: usize = 16; // 候補の数
const R: usize = 256; // selector_rank
const NB: usize = VOCAB.div_ceil(1024); // sel_top1 の区間の数 (gpu.cu の SEL_CH と同じ)
/// 出力層と埋め込み表の行数。語彙は VOCAB までで、残りは埋め草。候補は VOCAB の中から選ぶ (sglang_inkling_dflash2.py の valid_vocab_size)
const LV: usize = 201024;

struct FLayer {
    in_ln: P,
    post_ln: P,
    qn: P,
    kn: P,
    kp_attn: Fp8,
    kp_mlp: Fp8,
    base_attn: P, // [2][2][D] f32
    base_mlp: P,
    qkv: Fp8,
    wo: Fp8,
    gu: Fp8,
    down: Fp8,
    kc: P,
    vc: P,
}

impl FLayer {
    /// qkv のうち k と v の行 (文脈の K/V を作る)
    fn kv(&self) -> Fp8 {
        let q = NQ * HD;
        Fp8 { w: self.qkv.w + (q * D) as u64, s: self.qkv.s + f4(q), rows: 2048, cols: D }
    }
}

pub struct DFlash2 {
    layers: Vec<FLayer>,
    fc: Fp8,
    hidden_norm: P,
    norm: P,
    hproj: Fp8,
    pred: P, // [LV][R] BF16
    succ: P,
    mask: Vec<f32>,
    pub(super) ctx_end: usize,
    // 確信度の補正 (DSpark と同じ): (送った下書きの確率の積の和, 実際に当たった数) の指数移動平均
    cal: (f32, f32),
    sent: Option<(f32, usize)>,
    x: P,
    h: P,
    hc: P,
    dynk: P,
    qkv: P,
    att: P,
    o: P,
    oc: P,
    gu: P,
    act: P,
    hidden: P,
    logits: P,
    cat: P,
    ctx: P,
    kv: P,
    part: P,
    hp: P,
    pv: P,
    pi: P,
    pz: P,
    cv: P,
    ci: P,
    zr: P,
    res: P,
}

impl DFlash2 {
    pub fn load(g: &Gpu, path: &Path) -> Self {
        let st: &'static SafeTensors = Box::leak(Box::new(SafeTensors::open(path).expect("dflash2.safetensors を開けない")));
        let get = |name: &str, shape: &[usize]| -> &'static [u16] {
            let t = st.get(name).unwrap_or_else(|| panic!("{name} がない"));
            assert_eq!((t.dtype.as_str(), t.shape.as_slice()), ("BF16", shape), "{name}");
            model::cast(t.data)
        };
        let bfv = |name: &str, shape: &[usize]| model::bf_vec(get(name, shape));
        let mut up = Uploader::new(&g.stream);
        let layers = (0..L)
            .map(|l| {
                let p = |s: &str| format!("layers.{l}.{s}");
                let v = |s: &str, sh: &[usize], up: &mut Uploader| up.bytes(as_bytes(&bfv(&p(s), sh)));
                FLayer {
                    in_ln: v("input_layernorm.weight", &[D], &mut up),
                    post_ln: v("post_attention_layernorm.weight", &[D], &mut up),
                    qn: v("self_attn.q_norm.weight", &[HD], &mut up),
                    kn: v("self_attn.k_norm.weight", &[HD], &mut up),
                    kp_attn: fp8(g, &mut up, &[get(&p("attention_conv.kernel_projection.weight"), &[KP, D])], D),
                    kp_mlp: fp8(g, &mut up, &[get(&p("mlp_conv.kernel_projection.weight"), &[KP, D])], D),
                    base_attn: v("attention_conv.base_kernel", &[2, 2, D], &mut up),
                    base_mlp: v("mlp_conv.base_kernel", &[2, 2, D], &mut up),
                    qkv: fp8(
                        g,
                        &mut up,
                        &[
                            get(&p("self_attn.q_proj.weight"), &[NQ * HD, D]),
                            get(&p("self_attn.k_proj.weight"), &[1024, D]),
                            get(&p("self_attn.v_proj.weight"), &[1024, D]),
                        ],
                        D,
                    ),
                    wo: fp8(g, &mut up, &[get(&p("self_attn.o_proj.weight"), &[D, NQ * HD])], NQ * HD),
                    gu: fp8(g, &mut up, &[get(&p("mlp.gate_proj.weight"), &[FF, D]), get(&p("mlp.up_proj.weight"), &[FF, D])], D),
                    down: fp8(g, &mut up, &[get(&p("mlp.down_proj.weight"), &[D, FF])], FF),
                    kc: up.zeros(WIN * 2048),
                    vc: up.zeros(WIN * 2048),
                }
            })
            .collect();
        let mut mask = model::bf_row(g.cpu.embed, MASK, D);
        model::rmsnorm(&mut mask, Some(g.cpu.embed_norm));
        let fz = |n: usize, up: &mut Uploader| up.zeros(n * 4);
        let d = Self {
            layers,
            fc: fp8(g, &mut up, &[get("fc.weight", &[D, 4 * D])], 4 * D),
            hidden_norm: up.bytes(as_bytes(&bfv("hidden_norm.weight", &[D]))),
            norm: up.bytes(as_bytes(&bfv("norm.weight", &[D]))),
            hproj: fp8(g, &mut up, &[get("candidate_selector.hidden_projection.weight", &[R, D])], D),
            pred: up.bytes(as_bytes(get("candidate_selector.predecessor_codebook", &[LV, R]))),
            succ: up.bytes(as_bytes(get("candidate_selector.successor_codebook", &[LV, R]))),
            mask,
            ctx_end: 0,
            cal: (1.0, 1.0),
            sent: None,
            x: fz(B * D, &mut up),
            h: fz(B * D, &mut up),
            hc: fz(B * D, &mut up),
            dynk: fz(B * KP, &mut up),
            qkv: fz(B * QKV, &mut up),
            att: fz(B * NQ * HD, &mut up),
            o: fz(B * D, &mut up),
            oc: fz(B * D, &mut up),
            gu: fz(B * 2 * FF, &mut up),
            act: fz(B * FF, &mut up),
            hidden: fz(B * D, &mut up),
            logits: fz((B - 1) * LV, &mut up),
            cat: fz(B * 4 * D, &mut up),
            ctx: fz(B * D, &mut up),
            kv: fz(B * 2048, &mut up),
            part: fz((WIN + B).div_ceil(DR_SPLIT) * 8 * NQ * (HD + 2), &mut up),
            hp: fz((B - 1) * R, &mut up),
            pv: fz((B - 1) * NB * K, &mut up),
            pi: fz((B - 1) * NB * K, &mut up),
            pz: fz((B - 1) * NB, &mut up),
            cv: fz((B - 1) * K, &mut up),
            ci: fz((B - 1) * K, &mut up),
            zr: fz(B - 1, &mut up),
            res: fz(2 * (B - 1), &mut up),
        };
        g.stream.synchronize().unwrap();
        let (free, _) = result::mem_get_info().unwrap();
        eprintln!("下書きモデル (DFlash2) に {:.2} GB 載せた。残り {:.2} GB", up.total as f64 / 1e9, free as f64 / 1e9);
        d
    }

    fn mv(g: &Gpu, m: &Fp8, x: P, y: P, t: usize) {
        k!(g, "mv_fp8", m.rows.div_ceil(32), 256, 0; m.w, m.s, x, y, m.rows as i32, m.cols as i32, t as i32);
    }

    fn rms(g: &Gpu, x: P, w: P, y: P, t: usize) {
        k!(g, "rmsnorm", t, 256, 0; x, w, y, 0i32, t as i32);
    }

    fn conv(g: &Gpu, x: P, dynk: P, base: P, side: i32, y: P) {
        k!(g, "dconv", (B * D).div_ceil(256), 256, 0; x, dynk, base, side, y);
    }

    pub fn reset(&mut self) {
        self.ctx_end = 0;
        (self.cal, self.sent) = ((1.0, 1.0), None);
    }

    /// 退避と復元で丸ごと写す場所 (場所, バイト数)
    pub(super) fn kv_regions(&self) -> Vec<(P, usize)> {
        self.layers.iter().flat_map(|l| [(l.kc, WIN * 2048), (l.vc, WIN * 2048)]).collect()
    }

    /// 本体が今処理した n トークン (g.taps の行 0..n) を、位置 ctx_end から文脈に加える。
    /// 注意は直近 WIN 位置しか見ないので、それより古い行は計算しない
    pub fn add_ctx(&mut self, g: &Gpu, n: usize) {
        if let Some((p, k)) = self.sent.take() {
            let a = n.saturating_sub(1).min(k) as f32;
            self.cal = (0.9 * self.cal.0 + 0.1 * p, 0.9 * self.cal.1 + 0.1 * a);
        }
        let (taps, cap) = g.taps(n);
        for c0 in (n.saturating_sub(WIN)..n).step_by(B) {
            let m = (n - c0).min(B);
            k!(g, "tapcat", (m * 4 * D).div_ceil(256), 256, 0; taps + f4(c0 * D), (cap * D) as i32, m as i32, self.cat);
            Self::mv(g, &self.fc, self.cat, self.ctx, m);
            Self::rms(g, self.ctx, self.hidden_norm, self.ctx, m);
            for l in &self.layers {
                Self::mv(g, &l.kv(), self.ctx, self.kv, m);
                k!(g, "dr_ctx_store", (8, m), 128, 0; self.kv, l.kn, (self.ctx_end + c0) as i32, WIN as i32, l.kc, l.vc, THETA);
            }
        }
        self.ctx_end += n;
    }

    /// 本体が選んだ次のトークン next (文脈の続き) から、next の後に続くトークンを推測する。
    /// 選んだ候補の確率の積が tau を下回るか、max 個になったら止める
    pub fn draft(&mut self, g: &Gpu, next: u32, max: usize) -> Vec<u32> {
        if max == 0 {
            return vec![];
        }
        let tau = tau() * (self.cal.0 / self.cal.1.max(1e-3)).clamp(0.5, 2.0);
        let p = self.ctx_end;
        let mut xb = model::bf_row(g.cpu.embed, next as usize, D);
        model::rmsnorm(&mut xb, Some(g.cpu.embed_norm));
        for _ in 1..B {
            xb.extend_from_slice(&self.mask);
        }
        g.htod(self.x, &xb);
        let lo = p.saturating_sub(WIN - 1);
        let ns = (p - lo + B).div_ceil(DR_SPLIT);
        for l in &self.layers {
            Self::rms(g, self.x, l.in_ln, self.h, B);
            Self::mv(g, &l.kp_attn, self.h, self.dynk, B);
            Self::conv(g, self.h, self.dynk, l.base_attn, 0, self.hc);
            Self::mv(g, &l.qkv, self.hc, self.qkv, B);
            k!(g, "dr_qk", (NQ + 8, B), 128, 0; self.qkv, l.qn, l.kn, p as i32, G as i32, THETA);
            k!(g, "dr_attn2", (8, ns * 2), 128, 0; self.qkv, l.kc, l.vc, WIN as i32, lo as i32, p as i32, self.part, G as i32, WIN as i32);
            k!(g, "dr_fin", (NQ, B), 128, 0; self.part, ns as i32, self.att, G as i32);
            Self::mv(g, &l.wo, self.att, self.o, B);
            Self::conv(g, self.o, self.dynk, l.base_attn, 1, self.oc);
            k!(g, "add", (B * D).div_ceil(256), 256, 0; self.x, self.oc, self.x, (B * D) as i32);
            Self::rms(g, self.x, l.post_ln, self.h, B);
            Self::mv(g, &l.kp_mlp, self.h, self.dynk, B);
            Self::conv(g, self.h, self.dynk, l.base_mlp, 0, self.hc);
            Self::mv(g, &l.gu, self.hc, self.gu, B);
            k!(g, "silu_mul", (B * FF).div_ceil(256), 256, 0; self.gu, self.act, B as i32, FF as i32);
            Self::mv(g, &l.down, self.act, self.o, B);
            Self::conv(g, self.o, self.dynk, l.base_mlp, 1, self.oc);
            k!(g, "add", (B * D).div_ceil(256), 256, 0; self.x, self.oc, self.x, (B * D) as i32);
        }
        Self::rms(g, self.x, self.norm, self.hidden, B);
        // 行 0 は使わない
        let h1 = self.hidden + f4(D);
        g.logits(h1, self.logits, B - 1);
        Self::mv(g, &self.hproj, h1, self.hp, B - 1);
        k!(g, "sel_top1", (NB, B - 1), 256, 0; self.logits, LV as i32, VOCAB as i32, self.pv, self.pi, self.pz);
        k!(g, "sel_top2", B - 1, 256, 0; self.pv, self.pi, self.pz, NB as i32, self.cv, self.ci, self.zr);
        k!(g, "sel_walk", 1, 256, 0; self.cv, self.ci, self.zr, self.hp, self.pred, self.succ, next as i32, (B - 1) as i32, self.res);
        let r: Vec<u32> = g.dtoh(self.res, 2 * (B - 1));
        let (mut cum, mut ps) = (1f32, 0f32);
        let mut out = vec![];
        for i in 0..B - 1 {
            cum *= f32::from_bits(r[2 * i + 1]);
            if cum < tau || out.len() == max {
                break;
            }
            out.push(r[2 * i]);
            ps += cum;
        }
        self.sent = Some((ps, out.len()));
        out
    }
}
