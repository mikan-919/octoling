//! 実験: トレリス量子化した expert (experts.tq) を読み、CPU の順伝播で使う。OCTO_TQ=ファイル のときだけ読む。
//! 量子化は、入力側の回転 (W ≈ Q R^T)、LDLQ、ビットシフト型のトレリス (状態 L ビット、1 段で K ビット入れて V 個の値) で、
//! 値は表を使わずに状態から計算する (QTIP の 1MAD に倣う): 値 q = (4 バイトの和 (A_q * 状態 + B_q mod 2^32) - 510) / 147.8 (標準偏差がほぼ 1)。
//! 重み = 値 * 行のスケール。
//!
//! 形式 (リトルエンディアン):
//! - ヘッダ 64 バイト: "OCTQ0001", L (u32), 項目数 (u32、24 * 128 * 2), 表の位置 (u64), 回転 D の位置 (u64), 回転 FF の位置 (u64)
//! - 回転: f32 [c][c]。y = Q (R^T x)
//! - 表: 項目 (層 * 128 + expert) * 2 + (0: gate/up, 1: down) ごとに K (u8), V (u8), 0 (u16), 1 行の語数 (u32), ビット列の位置 (u64), スケールの位置 (u64)。
//!   K = 0 は量子化していない項目で、.strata の expert をそのまま使う
//! - ビット列: u32 [行][語数]。行の状態 i はビット列の [K * i, K * i + L) の窓 (下位ビットが先)
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
    /// 行 r と、回転した入力 x の内積
    pub(crate) fn dot(&self, r: usize, x: &[f32]) -> f32 {
        let w = &self.bits[r * self.ws..][..self.ws];
        let mask = (1u64 << self.l) - 1;
        let mut acc = 0f32;
        for (i, xs) in x.chunks_exact(self.v).enumerate() {
            let pos = self.k * i;
            let s = (((w[pos / 32] as u64 | (w[pos / 32 + 1] as u64) << 32) >> (pos % 32)) & mask) as u32;
            for (q, &xq) in xs.iter().enumerate() {
                let h = A[q].wrapping_mul(s).wrapping_add(B[q]);
                let bs = (h & 255) + (h >> 8 & 255) + (h >> 16 & 255) + (h >> 24);
                acc += (bs as f32 - 510.0) * xq;
            }
        }
        acc * (self.scale[r] / 147.8)
    }
}

pub(crate) struct Tq {
    mats: Vec<Option<TqMat>>,
    rot_d: Vec<f32>,  // R^T [D][D]
    rot_ff: Vec<f32>, // R^T [FF][FF]
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
        let transpose = |r: &[f32], c: usize| (0..c * c).map(|i| r[i % c * c + i / c]).collect::<Vec<f32>>();
        let (rot_d, rot_ff) = (transpose(f32s(u64_at(24), D * D), D), transpose(f32s(u64_at(32), FF * FF), FF));
        let mats = (0..n)
            .map(|i| {
                let e = &map[tab + i * 24..][..24];
                let (k, v) = (e[0] as usize, e[1] as usize);
                if k == 0 {
                    return None;
                }
                let ws = u32::from_le_bytes(e[4..8].try_into().unwrap()) as usize;
                let (bo, so) = (u64::from_le_bytes(e[8..16].try_into().unwrap()) as usize, u64::from_le_bytes(e[16..24].try_into().unwrap()) as usize);
                let rows = if i % 2 == 0 { 2 * FF } else { D };
                // SAFETY: 位置は 64 バイト境界で、範囲はファイルの中
                let bits = unsafe { std::slice::from_raw_parts(map[bo..bo + rows * ws * 4].as_ptr().cast(), rows * ws) };
                Some(TqMat { k, v, l, ws, bits, scale: f32s(so, rows) })
            })
            .collect();
        Ok(Self { mats, rot_d, rot_ff })
    }

    /// 層 il の expert ex の行列 (0: gate/up, 1: down)。量子化していなければ None
    pub(crate) fn mat(&self, il: usize, ex: usize, m: usize) -> Option<&TqMat> {
        self.mats[(il * N_EXP + ex) * 2 + m].as_ref()
    }

    /// R^T x (x の長さは D か FF)
    pub(crate) fn rotate(&self, x: &[f32]) -> Vec<f32> {
        let r = if x.len() == D { &self.rot_d } else { &self.rot_ff };
        let mut out = vec![0.0; x.len()];
        crate::pool::chunks_mut(&mut out, 64, |c, o| {
            for (i, o) in o.iter_mut().enumerate() {
                *o = r[(c * 64 + i) * x.len()..][..x.len()].iter().zip(x).map(|(a, b)| a * b).sum();
            }
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mad(s: u32, q: usize) -> f32 {
        let h = A[q].wrapping_mul(s).wrapping_add(B[q]);
        (h.to_le_bytes().iter().map(|&b| b as f32).sum::<f32>() - 510.0) / 147.8
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
