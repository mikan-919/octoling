//! トレリス量子化した expert (experts.tq) を読む。.strata と同じディレクトリにあれば、.strata の S2_K/S3_K の expert の代わりに使う。
//! 量子化は、入力側の回転 (W ≈ Q R^T)、LDLQ、ビットシフト型のトレリス (状態 L ビット、1 段で K ビット入れて V 個の値) で、
//! 値は表を使わずに状態から計算する (QTIP の 1MAD に倣う): 値 q = (4 バイトの和 (A_q * 状態 + B_q mod 2^32) - 510) / 147.8 (標準偏差がほぼ 1)。
//! 重み = 値 * 行のスケール。
//!
//! 形式 (リトルエンディアン):
//! - ヘッダ 64 バイト: "OCTQ0001", L (u32), 項目数 (u32、24 * 128 * 2), 表の位置 (u64), 回転 D の位置 (u64), 回転 FF の位置 (u64)
//! - 回転: f32 [c][c]。y = Q (R^T x)。R = diag(符号) (Qb ⊗ H128) (H128 は 128 次の正規化したアダマール行列、Qb は 128 のブロックをまたぐ直交行列) なので、
//!   読むときに符号と Qb に分け、R^T x は 128 ごとの高速アダマール変換とブロックの混ぜ合わせで計算する
//! - 表: 項目 (層 * 128 + expert) * 2 + (0: gate/up, 1: down) ごとに K (u8), V (u8), 0 (u16), 1 行の語数 (u32), ビット列の位置 (u64), スケールの位置 (u64)。
//!   K = 0 は量子化していない項目で、.strata の expert をそのまま使う
//! - ビット列: u32 [行][語数]。行の状態 i はビット列の [K * i, K * i + L) の窓 (下位ビットが先)。窓が行の終わりを越えたら、行の先頭に戻る。
//!   語数は K * 列 / V / 32 (前の形は行の終わりに詰め物があり、窓は行の中で終わる)
//! - スケール: f32 [行] (/ 147.8 を掛ける前の値)

use crate::model::{D, FF, N_EXP, N_LAYER};
use memmap2::Mmap;
use std::path::Path;

const A: [u32; 3] = [34038481, 2654435761, 3266489917];
const B: [u32; 3] = [76625530, 40503, 668265263];

pub(crate) struct TqMat {
    k: usize,
    v: usize,
    l: u32,
    ws: usize,
    bits: &'static [u32],
    scale: &'static [f32],
}

impl TqMat {
    /// GPU の SkMat の形式の番号 (16 | K << 8 | V << 12 | 語数 << 16)、ビット列、スケール (/ 147.8 を掛ける前)
    pub(crate) fn gpu(&self) -> (u32, &'static [u32], &'static [f32]) {
        (16 | (self.k as u32) << 8 | (self.v as u32) << 12 | (self.ws as u32) << 16, self.bits, self.scale)
    }

    /// 行 r と、回転した入力 x の内積
    pub(crate) fn dot(&self, r: usize, x: &[f32]) -> f32 {
        let w = &self.bits[r * self.ws..][..self.ws];
        let mask = (1u64 << self.l) - 1;
        let mut acc = 0f32;
        for (i, xs) in x.chunks_exact(self.v).enumerate() {
            let pos = self.k * i;
            let s = (((w[pos / 32] as u64 | (w[(pos / 32 + 1) % self.ws] as u64) << 32) >> (pos % 32)) & mask) as u32;
            for (q, &xq) in xs.iter().enumerate() {
                let h = A[q].wrapping_mul(s).wrapping_add(B[q]);
                let bs = (h & 255) + (h >> 8 & 255) + (h >> 16 & 255) + (h >> 24);
                acc += (bs as f32 - 510.0) * xq;
            }
        }
        acc * (self.scale[r] / 147.8)
    }
}

/// 入力側の回転 R = diag(sg) (q ⊗ H128)
pub(crate) struct Rot {
    pub(crate) sg: Vec<f32>, // [c]
    pub(crate) q: Vec<f32>,  // [c/128][c/128]
}

impl Rot {
    /// ファイルの R [c][c] を符号と q に分け、分けたもので R を組み直して元と合うか確かめる
    fn factor(r: &[f32], c: usize) -> Option<Self> {
        let nb = c / 128;
        let h = |j: usize, k: usize| (if (j & k).count_ones().is_multiple_of(2) { 1.0 } else { -1.0 }) / 128f32.sqrt();
        // R[i][b' * 128 + k] = sg[i] q[i / 128][b'] H[i % 128][k]。k = 0 の列から、ブロックの先頭の行の符号を + として読む
        let q: Vec<f32> = (0..nb * nb).map(|i| r[i / nb * 128 * c + i % nb * 128] * 128f32.sqrt()).collect();
        let sg: Vec<f32> = (0..c).map(|i| (r[i * c] * 128f32.sqrt() / q[i / 128 * nb]).signum()).collect();
        let ok = (0..c * c).all(|x| {
            let (i, j) = (x / c, x % c);
            (r[x] - sg[i] * q[i / 128 * nb + j / 128] * h(i % 128, j % 128)).abs() < 1e-5
        });
        ok.then_some(Self { sg, q })
    }

    /// R^T x
    pub(crate) fn apply(&self, x: &[f32]) -> Vec<f32> {
        let (c, nb) = (x.len(), x.len() / 128);
        let mut z: Vec<f32> = x.iter().zip(&self.sg).map(|(a, b)| a * b).collect();
        for b in z.chunks_mut(128) {
            let mut h = 1;
            while h < 128 {
                for i in (0..128).filter(|i| i & h == 0) {
                    let (u, v) = (b[i], b[i + h]);
                    (b[i], b[i + h]) = (u + v, u - v);
                }
                h *= 2;
            }
        }
        let n = 1.0 / 128f32.sqrt();
        (0..c).map(|j| (0..nb).map(|b| self.q[b * nb + j / 128] * z[b * 128 + j % 128]).sum::<f32>() * n).collect()
    }
}

pub(crate) struct Tq {
    mats: Vec<Option<TqMat>>,
    pub(crate) rot_d: Rot,
    pub(crate) rot_ff: Rot,
}

impl Tq {
    pub(crate) fn open(path: &Path) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        // SAFETY: 重みファイルは読み取り専用で、実行中に書き換えられない前提
        let map: &'static Mmap = Box::leak(Box::new(unsafe { Mmap::map(&file)? }));
        let bad = |m: &str| std::io::Error::other(format!("{}: {m}", path.display()));
        if map.len() < 64 || &map[..8] != b"OCTQ0001" {
            return Err(bad("experts.tq ではない"));
        }
        let u32_at = |o: usize| u32::from_le_bytes(map[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(map[o..o + 8].try_into().unwrap()) as usize;
        let (l, n, tab) = (u32_at(8), u32_at(12) as usize, u64_at(16));
        if n != N_LAYER * N_EXP * 2 || tab + n * 24 > map.len() {
            return Err(bad("表が合わない"));
        }
        let f32s = |o: usize, n: usize| -> &'static [f32] {
            // SAFETY: 位置は 64 バイト境界で、範囲はファイルの中 (下で確かめる)
            unsafe { std::slice::from_raw_parts(map[o..o + n * 4].as_ptr().cast(), n) }
        };
        if l != 12 {
            return Err(bad("状態は 12 ビットのみ対応"));
        }
        let rot = |o: usize, c: usize| Rot::factor(f32s(o, c * c), c).ok_or_else(|| bad("回転が符号 x (直交行列 ⊗ アダマール) の形でない"));
        let (rot_d, rot_ff) = (rot(u64_at(24), D)?, rot(u64_at(32), FF)?);
        let mats = (0..n)
            .map(|i| {
                let e = &map[tab + i * 24..][..24];
                let (k, v) = (e[0] as usize, e[1] as usize);
                if k == 0 {
                    return None;
                }
                let ws = u32::from_le_bytes(e[4..8].try_into().unwrap()) as usize;
                let (bo, so) = (u64::from_le_bytes(e[8..16].try_into().unwrap()) as usize, u64::from_le_bytes(e[16..24].try_into().unwrap()) as usize);
                if bo % 4 != 0 || so % 4 != 0 {
                    return Some(Err(bad("ビット列の位置が 4 の倍数でない")));
                }
                let (rows, cols) = if i % 2 == 0 { (2 * FF, D) } else { (D, FF) };
                if !(1..=5).contains(&k) || !(1..=3).contains(&v) || ws * 32 < k * cols / v || ws >= 1 << 16 {
                    return Some(Err(bad(&format!("項目 {i} の形式 (K {k}, V {v}, 語数 {ws}) を読めない"))));
                }
                // SAFETY: 位置は 64 バイト境界で、範囲はファイルの中
                let bits = unsafe { std::slice::from_raw_parts(map[bo..bo + rows * ws * 4].as_ptr().cast(), rows * ws) };
                Some(Ok(TqMat { k, v, l, ws, bits, scale: f32s(so, rows) }))
            })
            .map(Option::transpose)
            .collect::<Result<_, _>>()?;
        Ok(Self { mats, rot_d, rot_ff })
    }

    /// 層 il の expert ex の行列 (0: gate/up, 1: down)。量子化していなければ None
    pub(crate) fn mat(&self, il: usize, ex: usize, m: usize) -> Option<&TqMat> {
        self.mats[(il * N_EXP + ex) * 2 + m].as_ref()
    }

    /// R^T x (x の長さは D か FF)
    pub(crate) fn rotate(&self, x: &[f32]) -> Vec<f32> {
        if x.len() == D { &self.rot_d } else { &self.rot_ff }.apply(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mad(s: u32, q: usize) -> f32 {
        let h = A[q].wrapping_mul(s).wrapping_add(B[q]);
        (h.to_le_bytes().iter().map(|&b| b as f32).sum::<f32>() - 510.0) / 147.8
    }

    /// 符号 x (直交行列 ⊗ アダマール) で組んだ R を分けられ、apply が R^T x と同じになるか
    #[test]
    fn rot_factors_and_applies() {
        let c = 256;
        let (co, si) = (0.6f32, 0.8f32);
        let q = [-co, si, si, co]; // ブロックの先頭の行の符号が - の場合も含める
        let sg: Vec<f32> = (0..c).map(|i| if (i * 7 + 3) % 5 < 2 { -1.0 } else { 1.0 }).collect();
        let h = |j: usize, k: usize| (if (j & k).count_ones().is_multiple_of(2) { 1.0 } else { -1.0 }) / 128f32.sqrt();
        let r: Vec<f32> = (0..c * c).map(|x| sg[x / c] * q[x / c / 128 * 2 + x % c / 128] * h(x / c % 128, x % c % 128)).collect();
        let rot = Rot::factor(&r, c).expect("分けられない");
        let x: Vec<f32> = (0..c).map(|i| (i as f32 * 0.37).sin()).collect();
        let want: Vec<f32> = (0..c).map(|j| (0..c).map(|i| r[i * c + j] * x[i]).sum()).collect();
        let got = rot.apply(&x);
        assert!(got.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-4), "{:?} {:?}", &got[..4], &want[..4]);
    }

    /// 状態の列からビット列を組み (状態 0 の L ビット、続いて各段の上位 K ビット)、dot が状態ごとの値と x の内積になるか
    #[test]
    fn dot_reads_states_from_the_window() {
        let (l, k, v, n) = (12u32, 4usize, 3usize, 7usize);
        let mut states = vec![0xabcu32];
        for i in 1..n {
            let new = (i as u32 * 7 + 3) & ((1 << k) - 1);
            states.push((states[i - 1] >> k) | (new << (l as usize - k)));
        }
        let mut bits = vec![0u32; 8];
        let mut set = |pos: usize, b: u32| bits[pos / 32] |= (b & 1) << (pos % 32);
        (0..l as usize).for_each(|j| set(j, states[0] >> j));
        for (i, s) in states.iter().enumerate().skip(1) {
            (0..k).for_each(|j| set(l as usize + k * (i - 1) + j, s >> (l as usize - k + j)));
        }
        let x: Vec<f32> = (0..n * v).map(|i| i as f32 * 0.25 - 1.0).collect();
        let want: f32 = (0..n * v).map(|i| mad(states[i / v], i % v) * x[i]).sum::<f32>() * 0.5;
        let m = TqMat { k, v, l, ws: 8, bits: Box::leak(bits.into_boxed_slice()), scale: Box::leak(vec![0.5f32].into_boxed_slice()) };
        assert!((m.dot(0, &x) - want).abs() < 1e-3 * want.abs().max(1.0), "{} {want}", m.dot(0, &x));
    }
}
