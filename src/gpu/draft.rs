//! 下書きモデル DSpark (公式の mtp.safetensors)。投機的デコードで、本体が次に選ぶトークンの先を推測する。
//! 構造は Xolotl (MIT、Indras-Mirror/Xolotl の src/program/inkling_dspark.cpp) の説明と実装に合わせた:
//!   文脈: 本体の4つの層出力 (taps) を層ごとの重み (softmax(fuse_weight)) で混ぜ、正規化して文脈の K/V にする
//!   ブロック: 行 0 = 直前のトークンの埋め込み、行 1..7 = mask_embed。位置 P..P+7、3層の注意 (文脈 + ブロック、マスクなし) と MLP
//!   出力: 正規化した隠れ状態を本体の出力層に通したものを土台に、幅 256 の Markov ヘッドで1トークンずつ選ぶ
//! 重みは FP8 にして VRAM に置く。下書きは本体が必ず確かめるので、精度は受け入れ率にしか効かない。

use super::*;
use crate::st::SafeTensors;
use std::path::Path;

const B: usize = 8; // ブロックの行数
/// 下書きモデルが扱う文脈の長さ。global 層の KV を VRAM に置く (gpu.cu の DRAFT_CTX と同じ)
pub const DRAFT_CTX: usize = 8192;
const DRING: usize = 2048; // 局所層の文脈 K/V のリングバッファの行数
const LOCAL_WIN: usize = 1025; // 局所層が見る文脈の位置数
/// 下書きの top-1 確率の積がこれを下回ったら下書きを止める。OCTO_TAU で変えられる (既定 0.35 は Xolotl と同じ)
fn tau() -> f32 {
    static T: std::sync::LazyLock<f32> = std::sync::LazyLock::new(|| std::env::var("OCTO_TAU").ok().and_then(|s| s.parse().ok()).unwrap_or(0.35));
    *T
}
const R: usize = 256; // Markov ヘッドの幅
const HD: usize = 128;
const STAGE: usize = GROUP * (W13 + S13 + W2 + S2); // 置き場 1 つのバイト数

struct Fp8 {
    w: P,
    s: P,
    rows: usize,
    cols: usize,
}

struct DLayer {
    attn_norm: P,
    ffn_norm: P,
    dflare: P,
    qn: P,
    kn: P,
    knc: P,
    qkv: Fp8,
    kvc: Fp8,
    wo: Fp8,
    gu: Fp8,
    down: Fp8,
    kc: P,
    vc: P,
    ring: usize, // 0 なら global 層
}

pub struct Drafter {
    layers: Vec<DLayer>,
    norm: P,
    markov_norm: P,
    mask: Vec<f32>,
    alpha: [[f32; 4]; 4],
    jctx: P,
    jb: P,
    j01: Vec<f32>,      // [768][512]: rnn_proj の、状態と前のトークンに掛かる列
    w1: &'static [u16], // [201024][256]: 前のトークンの幅 256 の埋め込み
    fold: Fp8,          // [201024][256] = 本体の出力層 x hidden_out
    fold_b: P,
    ctx_end: usize,
    x: P,
    h: P,
    qkv: P,
    att: P,
    o: P,
    gu: P,
    act: P,
    hidden: P,
    base: P,
    x2: P,
    ctxm: P,
    frnn: P,
    mix: P,
    kv: P,
    lat: P,
    lg: P,
    res: P,
    anchor: P, // [4][D]: 文脈の最後の位置の taps
}

/// BF16 の行列 (parts を縦に並べたもの、列数 cols) を FP8 にして VRAM に置く。置き場 0 を経由する
fn fp8(g: &Gpu, up: &mut Uploader, parts: &[&'static [u16]], cols: usize) -> Fp8 {
    let rows: usize = parts.iter().map(|p| p.len() / cols).sum();
    let m = Fp8 { w: up.zeros(rows * cols), s: up.zeros(rows * 4), rows, cols };
    let chunk = STAGE / (cols * 2);
    let mut r0 = 0;
    for part in parts {
        for c in part.chunks(chunk * cols) {
            let n = c.len() / cols;
            up.to(g.staging[0], as_bytes(c));
            k!(g, "quant_rows_bf16", n, 256, 0; g.staging[0], cols as i32, m.w + (r0 * cols) as u64, m.s + f4(r0));
            r0 += n;
        }
    }
    g.stream.synchronize().unwrap();
    m
}

impl Drafter {
    pub fn load(g: &Gpu, path: &Path) -> Self {
        let st: &'static SafeTensors = Box::leak(Box::new(SafeTensors::open(path).expect("mtp.safetensors を開けない")));
        let get = |name: &str, shape: &[usize]| -> &'static [u16] {
            let t = st.get(name).unwrap_or_else(|| panic!("{name} がない"));
            assert_eq!((t.dtype.as_str(), t.shape.as_slice()), ("BF16", shape), "{name}");
            model::cast(t.data)
        };
        let bfv = |name: &str, shape: &[usize]| model::bf_vec(get(name, shape));
        let mut up = Uploader::new(&g.stream);

        let layers = (0..3)
            .map(|l| {
                let p = |s: &str| format!("layers.{l}.{s}");
                let v = |s: &str, n: usize, up: &mut Uploader| up.bytes(as_bytes(&bfv(&p(s), &[n])));
                let ring = if l == 1 { 0 } else { DRING };
                let rows = if ring == 0 { DRAFT_CTX } else { DRING };
                DLayer {
                    attn_norm: v("input_layernorm.weight", D, &mut up),
                    ffn_norm: v("post_attention_layernorm.weight", D, &mut up),
                    dflare: v("dflare_norm.weight", D, &mut up),
                    qn: v("self_attn.q_norm.weight", HD, &mut up),
                    kn: v("self_attn.k_norm.weight", HD, &mut up),
                    knc: v("self_attn.k_norm_ctx.weight", HD, &mut up),
                    qkv: fp8(g, &mut up, &[get(&p("self_attn.q_proj.weight"), &[3072, D]), get(&p("self_attn.k_proj.weight"), &[1024, D]), get(&p("self_attn.v_proj.weight"), &[1024, D])], D),
                    kvc: fp8(g, &mut up, &[get(&p("self_attn.k_proj_ctx.weight"), &[1024, D]), get(&p("self_attn.v_proj_ctx.weight"), &[1024, D])], D),
                    wo: fp8(g, &mut up, &[get(&p("self_attn.o_proj.weight"), &[D, 3072])], 3072),
                    gu: fp8(g, &mut up, &[get(&p("mlp.gate_proj.weight"), &[9216, D]), get(&p("mlp.up_proj.weight"), &[9216, D])], D),
                    down: fp8(g, &mut up, &[get(&p("mlp.down_proj.weight"), &[D, 9216])], 9216),
                    kc: up.zeros(rows * 1024 * 2),
                    vc: up.zeros(rows * 1024 * 2),
                    ring,
                }
            })
            .collect();

        // fold = 本体の出力層 x hidden_out ([201024][256])。語彙を 16384 行ずつに分けて行列積をとり、FP8 にする
        let ho = bfv("markov_head.hidden_out.weight", &[D, R]);
        let hot: Vec<f32> = (0..R).flat_map(|k| (0..D).map(move |d| (k, d))).map(|(k, d)| ho[d * R + k]).collect();
        up.to(g.staging[1], as_bytes(&hot));
        let fold = Fp8 { w: up.zeros(201024 * R), s: up.zeros(201024 * 4), rows: 201024, cols: R };
        for v0 in (0..201024).step_by(16384) {
            let rows = 16384.min(201024 - v0);
            g.gemm(g.unembed + (v0 * D * 2) as u64, g.staging[1], g.staging[0], rows, D, R);
            k!(g, "quant_cols_f32", rows, 256, 0; g.staging[0], rows as i32, R as i32, fold.w + (v0 * R) as u64, fold.s + f4(v0));
        }
        let hb = up.bytes(as_bytes(&bfv("markov_head.hidden_out.bias", &[D])));
        let fold_b = up.zeros(201024 * 4);
        g.gemm(g.unembed, hb, fold_b, 201024, D, 1);
        g.stream.synchronize().unwrap();

        let rnn = get("markov_head.rnn_proj.weight", &[768, 5120]);
        let j01 = (0..768).flat_map(|o| model::bf_vec(&rnn[o * 5120..][..512])).collect();
        let jctx: Vec<u16> = (0..768).flat_map(|o| rnn[o * 5120 + 512..][..4608].iter().copied()).collect();
        let fuse = bfv("fuse_weight", &[4, 4]);
        let alpha = std::array::from_fn(|r| {
            let row = &fuse[r * 4..][..4];
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = row.iter().map(|x| (x - m).exp()).collect();
            let z: f32 = e.iter().sum();
            std::array::from_fn(|j| e[j] / z)
        });
        let fz = |n: usize, up: &mut Uploader| up.zeros(n * 4);
        let d = Self {
            layers,
            norm: up.bytes(as_bytes(&bfv("norm.weight", &[D]))),
            markov_norm: up.bytes(as_bytes(&bfv("markov_head.dflare_norm.weight", &[D]))),
            mask: bfv("mask_embed", &[D]),
            alpha,
            jctx: up.bytes(as_bytes(&jctx)),
            jb: up.bytes(as_bytes(&bfv("markov_head.rnn_proj.bias", &[768]))),
            j01,
            w1: get("markov_head.markov_w1.weight", &[201024, R]),
            fold,
            fold_b,
            ctx_end: 0,
            x: fz(B * D, &mut up),
            h: fz(B * D, &mut up),
            qkv: fz(B * 5120, &mut up),
            att: fz(B * 3072, &mut up),
            o: fz(B * D, &mut up),
            gu: fz(B * 2 * 9216, &mut up),
            act: fz(B * 9216, &mut up),
            hidden: fz(B * D, &mut up),
            base: fz(B * 201024, &mut up),
            x2: fz(B * 2 * D, &mut up),
            ctxm: fz(B * 768, &mut up),
            frnn: fz(D, &mut up),
            mix: fz(B * D, &mut up),
            kv: fz(B * 2048, &mut up),
            lat: fz(R, &mut up),
            lg: fz(201024, &mut up),
            res: fz(2, &mut up),
            anchor: fz(4 * D, &mut up),
        };
        let (free, _) = result::mem_get_info().unwrap();
        eprintln!("下書きモデルに {:.2} GB 載せた。残り {:.2} GB", up.total as f64 / 1e9, free as f64 / 1e9);
        d
    }

    fn mv(&self, g: &Gpu, m: &Fp8, x: P, y: P, t: usize) {
        k!(g, "mv_fp8", (m.rows * 32).div_ceil(256), 256, 0; m.w, m.s, x, y, m.rows as i32, m.cols as i32, t as i32);
    }

    fn rms(g: &Gpu, x: P, w: P, y: P, t: usize) {
        k!(g, "rmsnorm", t, 256, 0; x, w, y, D as i32, 0i32, t as i32);
    }

    pub fn reset(&mut self) {
        self.ctx_end = 0;
    }

    /// 本体が今処理した n トークン (g.b.taps の行 0..n) を、位置 ctx_end から文脈に加える
    pub fn add_ctx(&mut self, g: &Gpu, n: usize) {
        for c0 in (0..n).step_by(B) {
            let m = (n - c0).min(B);
            for (li, l) in self.layers.iter().enumerate() {
                let a = self.alpha[li];
                k!(g, "tapmix", (m * D).div_ceil(256), 256, 0; g.b.taps + f4(c0 * D), (TMAX * D) as i32, a[0], a[1], a[2], a[3], m as i32, self.mix);
                Self::rms(g, self.mix, l.dflare, self.mix, m);
                self.mv(g, &l.kvc, self.mix, self.kv, m);
                k!(g, "dr_ctx_store", (8, m), 128, 0; self.kv, l.knc, (self.ctx_end + c0) as i32, l.ring as i32, l.kc, l.vc);
            }
        }
        for j in 0..4 {
            g.copy(self.anchor + f4(j * D), g.b.taps + f4(j * TMAX * D + (n - 1) * D), D);
        }
        self.ctx_end += n;
    }

    /// 文脈の最後のトークン anchor と、本体が選んだ次のトークン next から、next の後に続くトークンを推測する。
    /// 推測の top-1 確率の積が TAU を下回るか、max 個になったら止める
    pub fn draft(&mut self, g: &Gpu, anchor: u32, next: u32, max: usize) -> Vec<u32> {
        if max == 0 {
            return vec![];
        }
        let p = self.ctx_end;
        let mut xb = model::bf_row(g.cpu.embed, anchor as usize, D);
        model::rmsnorm(&mut xb, Some(g.cpu.embed_norm));
        for _ in 1..B {
            xb.extend_from_slice(&self.mask);
        }
        g.htod(self.x, &xb);
        for l in &self.layers {
            Self::rms(g, self.x, l.attn_norm, self.h, B);
            self.mv(g, &l.qkv, self.h, self.qkv, B);
            k!(g, "dr_qk", (32, B), 128, 0; self.qkv, l.qn, l.kn, p as i32);
            let lo = if l.ring > 0 { p.saturating_sub(LOCAL_WIN) } else { 0 };
            k!(g, "dr_attn", (24, B), 128, 0; self.qkv, l.kc, l.vc, l.ring as i32, lo as i32, p as i32, self.att);
            self.mv(g, &l.wo, self.att, self.o, B);
            k!(g, "add", (B * D).div_ceil(256), 256, 0; self.x, self.o, self.x, (B * D) as i32);
            Self::rms(g, self.x, l.ffn_norm, self.h, B);
            self.mv(g, &l.gu, self.h, self.gu, B);
            k!(g, "silu_mul", (B * 9216).div_ceil(256), 256, 0; self.gu, self.act, B as i32);
            self.mv(g, &l.down, self.act, self.o, B);
            k!(g, "add", (B * D).div_ceil(256), 256, 0; self.x, self.o, self.x, (B * D) as i32);
        }
        Self::rms(g, self.x, self.norm, self.hidden, B);
        g.gemm(g.unembed, self.hidden, self.base, 201024, D, B);
        let a = self.alpha[3];
        k!(g, "tapmix", D.div_ceil(256), 256, 0; self.anchor, D as i32, a[0], a[1], a[2], a[3], 1i32, self.frnn);
        Self::rms(g, self.frnn, self.markov_norm, self.frnn, 1);
        k!(g, "concat_frnn", (B * 2 * D).div_ceil(256), 256, 0; self.hidden, self.frnn, self.x2);
        g.gemm(self.jctx, self.x2, self.ctxm, 768, 2 * D, B);
        k!(g, "add_bias_rows", (B * 768).div_ceil(256), 256, 0; self.ctxm, self.jb, 768i32, (B * 768) as i32);
        let ctxm: Vec<f32> = g.dtoh(self.ctxm, B * 768);

        // Markov ヘッド: 行 i は位置 P-1+i のトークンを選ぶ。行 1 は next なので、状態だけ進めて next を前のトークンにする
        let (mut state, mut prev, mut cum) = ([0f32; R], anchor, 1f32);
        let mut out = vec![];
        for i in 1..B {
            let w1 = &self.w1[prev as usize * R..][..R];
            let mut raw = ctxm[i * 768..][..768].to_vec();
            for (o, r) in raw.iter_mut().enumerate() {
                let j = &self.j01[o * 512..][..512];
                *r += (0..R).map(|e| j[e] * state[e] + j[R + e] * f32::from_bits((w1[e] as u32) << 16)).sum::<f32>();
            }
            let mut latent = [0f32; R];
            for e in 0..R {
                let gate = 1.0 / (1.0 + (-raw[e]).exp());
                state[e] = gate * state[e] + (1.0 - gate) * raw[R + e].tanh();
                latent[e] = raw[2 * R + e].tanh();
            }
            if i == 1 {
                // OCTO_DRAFT_DEBUG: 行 1 の推測が next と合うかを数える (下書きモデルの実装の確かめ)
                if std::env::var_os("OCTO_DRAFT_DEBUG").is_some() {
                    g.htod(self.lat, &latent);
                    self.mv(g, &self.fold, self.lat, self.lg, 1);
                    k!(g, "fold_finish", 1, 1024, 0; self.lg, self.fold_b, self.base + f4(201024), VOCAB as i32, self.res, self.res + 4);
                    let r: Vec<u32> = g.dtoh(self.res, 2);
                    eprintln!("draft row1 {} next {next} {}", r[0], if r[0] == next { "ok" } else { "miss" });
                }
                prev = next;
                continue;
            }
            g.htod(self.lat, &latent);
            self.mv(g, &self.fold, self.lat, self.lg, 1);
            k!(g, "fold_finish", 1, 1024, 0; self.lg, self.fold_b, self.base + f4(i * 201024), VOCAB as i32, self.res, self.res + 4);
            let r: Vec<u32> = g.dtoh(self.res, 2);
            cum /= f32::from_bits(r[1]).max(1.0);
            if cum < tau() {
                break;
            }
            out.push(r[0]);
            prev = r[0];
            if out.len() == max {
                break;
            }
        }
        out
    }
}
