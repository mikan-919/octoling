//! Strata の量子化済み重み `.strata` v1.1 を読む。対応するのは SQ-10 (expert が S2_K, S3_K, SF4)。
//! 形式は Strata (MIT) の strata_reader.cpp, ink_manifest.h, sk_codecs.cpp に合わせている。
//! 本体の行列は BF16 に広げて公式の重みと同じカーネルで計算し、expert だけ専用のカーネルで計算する。

use crate::model::{D, Experts, FF, N_EXP, N_LAYER};
use memmap2::Mmap;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

const SF4: u8 = 15;
/// 1レコード (gate/up の1行、down の2行の組) の super-block (256 要素) の数
const SB_REC: usize = D / 256;
const GU_RECS: usize = 2 * FF;
const DN_RECS: usize = D / 2;

fn sb_bytes(code: u8) -> usize {
    match code {
        2 => 70,
        3 => 102,
        6 => 198,
        _ => panic!("未対応の S_K 形式 {code} (対応は SQ-10 のみ)"),
    }
}

fn f16(h: u16) -> f32 {
    let (s, e, m) = ((h as u32 & 0x8000) << 16, (h as u32 >> 10) & 31, h as u32 & 0x3ff);
    let mag = match e {
        0 => m as f32 * f32::powi(2.0, -24),
        31 => f32::from_bits(0x7f80_0000 | (m << 13)),
        _ => f32::from_bits(((e + 112) << 23) | (m << 13)),
    };
    f32::from_bits(mag.to_bits() | s)
}

fn to_bf16(f: f32) -> u16 {
    let u = f.to_bits();
    (u.wrapping_add(0x7fff).wrapping_add((u >> 16) & 1) >> 16) as u16
}

/// S_K の行 (out.len() は 256 の倍数) を f32 に展開する。w = (q - half) * D * u / 255
fn sk_row(code: u8, raw: &[u8], out: &mut [f32]) {
    let sb = sb_bytes(code);
    let half = match code {
        2 => 1.5,
        3 => 3.5,
        _ => 31.5,
    };
    for (s, o) in out.chunks_mut(256).enumerate() {
        let b = &raw[s * sb..][..sb];
        let d = f16(u16::from_le_bytes([b[0], b[1]]));
        let p = &b[6..];
        for (i, o) in o.iter_mut().enumerate() {
            let two = ((p[i / 4] >> (2 * (i % 4))) & 3) as u32;
            let q = match code {
                2 => two,
                3 => two | (((p[64 + i / 8] >> (i % 8)) & 1) as u32) << 2,
                _ => ((p[i / 2] >> (4 * (i % 2))) & 15) as u32 | (((p[128 + i / 4] >> (2 * (i % 4))) & 3) as u32) << 4,
            };
            *o = (q as f32 - half) * (d * (b[2 + i / 64] as f32 / 255.0));
        }
    }
}

fn leak_bytes<T: 'static>(v: Vec<T>) -> &'static [u8] {
    let s: &'static [T] = Box::leak(v.into_boxed_slice());
    // SAFETY: u16/f32 の配列をバイト列として読むだけ
    unsafe { std::slice::from_raw_parts(s.as_ptr().cast(), std::mem::size_of_val(s)) }
}

struct Tensor {
    fmt: String,
    shape: Vec<usize>,
    off: usize,
    bytes: usize,
}

pub struct Pack {
    sec1: &'static [u8],
    sec4: &'static [u8],
    trunk: HashMap<String, Tensor>,
    dir: Vec<(u8, u8, usize, usize, usize)>, // [層 * 128 + expert] の (gate/up の形式, down の形式, 本体の位置, 長さ, down の位置)
}

impl Pack {
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        // SAFETY: 重みファイルは読み取り専用で、実行中に書き換えられない前提
        let map: &'static Mmap = Box::leak(Box::new(unsafe { Mmap::map(&file)? }));
        let bad = |m: &str| std::io::Error::other(format!("{}: {m}", path.display()));
        if map.len() < 4096 || &map[..8] != b"STRATA\0\x01" {
            return Err(bad(".strata v1 ではない"));
        }
        let u64_at = |o: usize| u64::from_le_bytes(map[o..o + 8].try_into().unwrap()) as usize;
        let sec = |i: usize| -> std::io::Result<&'static [u8]> {
            let (o, n) = (u64_at(0x10 + i * 64), u64_at(0x18 + i * 64));
            map.get(o..o + n).ok_or_else(|| bad("セクションがファイルの外を指している"))
        };
        let mut text = vec![];
        ruzstd::decoding::StreamingDecoder::new(sec(0)?).map_err(|e| bad(&e.to_string()))?.read_to_end(&mut text)?;
        let man: Value = serde_json::from_slice(&text).map_err(|e| bad(&e.to_string()))?;
        let build = man["build"].as_str().unwrap_or("?");
        if build != "SQ-10" {
            return Err(bad(&format!("{build} には対応していない (対応は SQ-10 のみ)")));
        }
        let trunk = man["trunk_tensors"]
            .as_array()
            .ok_or_else(|| bad("trunk_tensors がない"))?
            .iter()
            .map(|t| {
                let n = |k: &str| t[k].as_u64().unwrap() as usize;
                let shape = t["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
                (t["name"].as_str().unwrap().to_owned(), Tensor { fmt: t["format"].as_str().unwrap().to_owned(), shape, off: n("offset"), bytes: n("bytes") })
            })
            .collect();
        let (s3, sec4) = (sec(3)?, sec(4)?);
        let n = N_LAYER * N_EXP;
        if s3.len() < n * 32 {
            return Err(bad("expert の目録が短い"));
        }
        let mut dir = vec![(0, 0, 0, 0, 0); n];
        for (i, e) in s3[..n * 32].chunks(32).enumerate() {
            let id = u16::from_le_bytes([e[0], e[1]]) as usize;
            let u32_at = |o: usize| u32::from_le_bytes(e[o..o + 4].try_into().unwrap()) as usize;
            let off = u64::from_le_bytes(e[8..16].try_into().unwrap()) as usize;
            let (gu, dn, len, dn_off) = (e[2], e[3], u32_at(16), u32_at(20));
            let ok = id < N_EXP && matches!((gu, dn), (2 | 3, 2 | 3) | (SF4, SF4)) && off + len <= sec4.len() && {
                let need = if gu == SF4 { dn_off + D * FF / 2 + D * FF / 16 } else { dn_off + DN_RECS * SB_REC * sb_bytes(dn) };
                need <= len
            };
            if !ok {
                return Err(bad(&format!("expert {i} の形式 ({gu}, {dn}) を読めない")));
            }
            dir[i / N_EXP * N_EXP + id] = (gu, dn, off, len, dn_off);
        }
        Ok(Self { sec1: sec(1)?, sec4, trunk, dir })
    }

    /// 名前のテンソル (列は D) が S6_K なら、GPU 向けに並べ直したもの [下位4ビット rows * D/2][上位2ビット rows * D/4][64 要素ごとのスケール D * u / 255 (f32) rows * D/64] を返す
    pub(crate) fn s6(&self, name: &str) -> Option<&'static [u8]> {
        let t = self.trunk.get(name)?;
        if t.fmt != "S6_K" {
            return None;
        }
        let raw = &self.sec1[t.off..t.off + t.bytes];
        let rows = t.shape[0];
        let mut out = vec![0u8; rows * (D / 2 + D / 4 + D / 64 * 4)];
        let (lo, rest) = out.split_at_mut(rows * D / 2);
        let (hi, sc) = rest.split_at_mut(rows * D / 4);
        let (lo, hi, sc) = (lo.as_mut_ptr() as usize, hi.as_mut_ptr() as usize, sc.as_mut_ptr() as usize);
        crate::pool::run(rows.div_ceil(256), &|c| {
            for r in c * 256..(c * 256 + 256).min(rows) {
                for s in 0..SB_REC {
                    let b = &raw[(r * SB_REC + s) * sb_bytes(6)..][..sb_bytes(6)];
                    let d = f16(u16::from_le_bytes([b[0], b[1]]));
                    // SAFETY: 行ごとに重ならない範囲に書く
                    unsafe {
                        std::ptr::copy_nonoverlapping(b[6..].as_ptr(), (lo + r * D / 2 + s * 128) as *mut u8, 128);
                        std::ptr::copy_nonoverlapping(b[134..].as_ptr(), (hi + r * D / 4 + s * 64) as *mut u8, 64);
                        for g in 0..4 {
                            *((sc + (r * D / 64 + s * 4 + g) * 4) as *mut f32) = d * b[2 + g] as f32 / 255.0;
                        }
                    }
                }
            }
        });
        Some(leak_bytes(out))
    }

    /// 名前のテンソルが Q8_ROW なら、その生のバイト列 ([行] の f32 スケール; int8 [行][列]) を返す
    pub(crate) fn q8(&self, name: &str) -> Option<&'static [u8]> {
        let t = self.trunk.get(name)?;
        (t.fmt == "Q8_ROW").then(|| &self.sec1[t.off..t.off + t.bytes])
    }

    /// 名前のテンソルを want (BF16 か F32) のバイト列で返す。形式が違えば展開する
    pub(crate) fn tensor(&self, name: &str, want: &str, shape: &[usize]) -> &'static [u8] {
        let t = self.trunk.get(name).unwrap_or_else(|| panic!("{name} がない"));
        assert_eq!(t.shape, shape, "{name}");
        let raw: &'static [u8] = &self.sec1[t.off..t.off + t.bytes];
        if t.fmt == want && (raw.as_ptr() as usize).is_multiple_of(4) {
            return raw;
        }
        let (rows, cols) = if t.fmt == "Q8_ROW" || t.fmt == "S6_K" { (shape[0], shape[1]) } else { (1, shape.iter().product()) };
        let row = |r: usize, o: &mut [f32]| match t.fmt.as_str() {
            "F32" => o.iter_mut().zip(raw.as_chunks::<4>().0).for_each(|(o, b)| *o = f32::from_le_bytes(*b)),
            "BF16" => o.iter_mut().zip(raw.as_chunks::<2>().0).for_each(|(o, b)| *o = f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)),
            "Q8_ROW" => {
                let sc = f32::from_le_bytes(raw[r * 4..r * 4 + 4].try_into().unwrap());
                let q = &raw[rows * 4 + r * cols..][..cols];
                o.iter_mut().zip(q).for_each(|(o, &q)| *o = q as i8 as f32 * sc);
            }
            "S6_K" => sk_row(6, &raw[r * SB_REC * sb_bytes(6)..], o),
            f => panic!("{name}: 未対応の形式 {f}"),
        };
        const CHUNK: usize = 256;
        match want {
            "BF16" => {
                let mut v = vec![0u16; rows * cols];
                crate::pool::chunks_mut(&mut v, cols * CHUNK, |c, o| {
                    let mut f = vec![0.0; cols];
                    for (i, o) in o.chunks_mut(cols).enumerate() {
                        row(c * CHUNK + i, &mut f);
                        o.iter_mut().zip(&f).for_each(|(o, &x)| *o = to_bf16(x));
                    }
                });
                leak_bytes(v)
            }
            _ => {
                let mut v = vec![0f32; rows * cols];
                crate::pool::chunks_mut(&mut v, cols * CHUNK, |c, o| {
                    for (i, o) in o.chunks_mut(cols).enumerate() {
                        row(c * CHUNK + i, o);
                    }
                });
                leak_bytes(v)
            }
        }
    }

    pub(crate) fn experts(&self, il: usize) -> Experts {
        let bodies = (0..N_EXP)
            .map(|e| {
                let (gu, dn, off, len, dn_off) = self.dir[il * N_EXP + e];
                SkBody { gu, dn, body: &self.sec4[off..off + len], dn_off }
            })
            .collect();
        Experts { mx: false, w13: &[], s13: &[], a13: &[], w2: &[], s2: &[], a2: &[], sk: Some(SkLayer { bodies }) }
    }
}

struct SkBody {
    gu: u8,
    dn: u8,
    body: &'static [u8],
    dn_off: usize,
}

impl SkBody {
    fn outer(&self) -> [f32; 2] {
        let h: Vec<f32> = self.body[..16].chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        [h[0] / h[1], h[2] / h[3]]
    }
}

/// 1層の expert 128 個
pub(crate) struct SkLayer {
    bodies: Vec<SkBody>,
}

/// GPU に置く1層分: 重みの平面、行列ごとの [lo の位置, scale の位置, 形式 | gain << 32] (image の先頭から)、活性の逆スケール [128][FF]
pub(crate) struct SkImage {
    pub(crate) bytes: Vec<u8>,
    pub(crate) tab: Vec<[u64; 3]>,
    pub(crate) ixd: Vec<f32>,
}

/// 行列1つの平面のバイト数 (lo, hi, scale)
fn planes(code: u8, recs: usize) -> [usize; 3] {
    match code {
        SF4 => [recs * 1152, 0, recs * 144],
        2 => [recs * 576, 0, recs * 54],
        _ => [recs * 576, recs * 288, recs * 54],
    }
}

/// GPU の expert の平面で、レコード rec の 16 要素のブロック blk の位置。4 ブロック (64 要素) の塊ごとに全レコードを並べる
/// ([塊][レコード][4])。GPU のワープは連続するレコードを受け持つので、1 回の読み込みが連続した領域になる
fn blk_at(recs: usize, rec: usize, blk: usize) -> usize {
    (blk / 4 * recs + rec) * 4 + blk % 4
}

/// S_K のレコード (super-block の並び) を、lo (2ビット x 16 の u32)、hi (3ビット目 x 16 の u16)、u [36][レコード] (u8)、
/// D [9][レコード] (u16) の平面に分ける。lo と hi の並びは blk_at
fn split_sk(code: u8, raw: &[u8], recs: usize, out: &mut [u8]) {
    let [lo_n, hi_n, _] = planes(code, recs);
    let (lo, rest) = out.split_at_mut(lo_n);
    let (hi, sc) = rest.split_at_mut(hi_n);
    let (u, d) = sc.split_at_mut(recs * 36);
    let sb = sb_bytes(code);
    for (i, b) in raw[..recs * SB_REC * sb].chunks_exact(sb).enumerate() {
        let (rec, s) = (i / SB_REC, i % SB_REC);
        d[(s * recs + rec) * 2..][..2].copy_from_slice(&b[..2]);
        for j in 0..4 {
            u[(s * 4 + j) * recs + rec] = b[2 + j];
        }
        for k in 0..16 {
            let at = blk_at(recs, rec, s * 16 + k);
            lo[at * 4..][..4].copy_from_slice(&b[6 + k * 4..][..4]);
            if code == 3 {
                hi[at * 2..][..2].copy_from_slice(&b[70 + k * 2..][..2]);
            }
        }
    }
}

/// SF4 の lo [レコード][144] (u64) と sc [レコード][144] (u8) を blk_at の並びにする
fn split_sf4(raw: &[u8], recs: usize, out: &mut [u8]) {
    let (lo, sc) = out.split_at_mut(recs * 1152);
    for rec in 0..recs {
        for blk in 0..144 {
            let at = blk_at(recs, rec, blk);
            lo[at * 8..][..8].copy_from_slice(&raw[(rec * 144 + blk) * 8..][..8]);
            sc[at] = raw[recs * 1152 + rec * 144 + blk];
        }
    }
}

impl SkLayer {
    /// 行 r (gate と up が交互) と x の内積
    pub(crate) fn up(&self, ex: usize, r: usize, x: &[f32]) -> f32 {
        let b = &self.bodies[ex];
        let mut w = [0.0; D];
        if b.gu == SF4 {
            crate::nvfp4::dequant_row(&b.body[16 + r * D / 2..][..D / 2], &b.body[16 + GU_RECS * D / 2 + r * D / 16..][..D / 16], b.outer()[0], &mut w);
        } else {
            sk_row(b.gu, &b.body[2 * FF + r * SB_REC * sb_bytes(b.gu)..], &mut w);
        }
        w.iter().zip(x).map(|(w, x)| w * x).sum()
    }

    /// down の行 r と活性 act の内積。S_K は活性を列ごとのスケールで割ってから掛ける
    pub(crate) fn down(&self, ex: usize, r: usize, act: &[f32]) -> f32 {
        let b = &self.bodies[ex];
        if b.dn == SF4 {
            let mut w = [0.0; FF];
            crate::nvfp4::dequant_row(
                &b.body[b.dn_off + r * FF / 2..][..FF / 2],
                &b.body[b.dn_off + D * FF / 2 + r * FF / 16..][..FF / 16],
                b.outer()[1],
                &mut w,
            );
            return w.iter().zip(act).map(|(w, x)| w * x).sum();
        }
        let mut pair = [0.0; D];
        sk_row(b.dn, &b.body[b.dn_off + r / 2 * SB_REC * sb_bytes(b.dn)..], &mut pair);
        pair[r % 2 * FF..][..FF].iter().zip(act).zip(b.body.as_chunks::<2>().0).map(|((w, x), s)| w * x / f16(u16::from_le_bytes([s[0], s[1]]))).sum()
    }

    pub(crate) fn image(&self) -> SkImage {
        let mut at = 0;
        let mut tab = vec![[0u64; 3]; 2 * N_EXP];
        let mut places = Vec::with_capacity(N_EXP);
        for (e, b) in self.bodies.iter().enumerate() {
            let start = at;
            let g = if b.gu == SF4 { b.outer() } else { [1.0; 2] };
            for (m, (code, recs)) in [(b.gu, GU_RECS), (b.dn, DN_RECS)].into_iter().enumerate() {
                let [lo, hi, sc] = planes(code, recs);
                tab[e * 2 + m] = [at as u64, (at + lo + hi) as u64, code as u64 | (g[m].to_bits() as u64) << 32];
                at += (lo + hi + sc).next_multiple_of(16); // moe_skb が 16 バイトずつ読む
            }
            places.push((start, at - start));
        }
        let mut bytes = vec![0u8; at];
        let base = bytes.as_mut_ptr() as usize;
        crate::pool::run(N_EXP, &|e| {
            let (b, (start, n)) = (&self.bodies[e], places[e]);
            // SAFETY: expert ごとの範囲は重ならず、bytes は run が返るまで生きている
            let out = unsafe { std::slice::from_raw_parts_mut((base + start) as *mut u8, n) };
            let g = planes(b.gu, GU_RECS).iter().sum::<usize>().next_multiple_of(16);
            let (gu, dn) = out.split_at_mut(g);
            if b.gu == SF4 {
                split_sf4(&b.body[16..], GU_RECS, gu);
                split_sf4(&b.body[b.dn_off..], DN_RECS, dn);
            } else {
                split_sk(b.gu, &b.body[2 * FF..], GU_RECS, gu);
                split_sk(b.dn, &b.body[b.dn_off..], DN_RECS, dn);
            }
        });
        let mut ixd = vec![1.0; N_EXP * FF];
        for (b, o) in self.bodies.iter().zip(ixd.chunks_mut(FF)) {
            if b.dn != SF4 {
                o.iter_mut().zip(b.body.as_chunks::<2>().0).for_each(|(o, s)| *o = 1.0 / f16(u16::from_le_bytes([s[0], s[1]])));
            }
        }
        SkImage { bytes, tab, ixd }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_s3_superblock() {
        let mut sb = vec![0u8; 102];
        sb[..2].copy_from_slice(&0x4000u16.to_le_bytes()); // D = 2.0
        sb[2] = 255; // 先頭 64 要素の u = 1.0
        sb[6] = 0b11; // q[0] の下位 2 ビット = 3
        sb[70] = 0b1; // q[0] の上位ビット (q[0] = 7)
        let mut o = vec![0.0; 256];
        sk_row(3, &sb, &mut o);
        assert_eq!(o[0], (7.0 - 3.5) * 2.0);
        assert_eq!(o[1], (0.0 - 3.5) * 2.0);
        assert_eq!(o[64], 0.0); // u = 0
    }

    #[test]
    fn f16_matches_known_values() {
        assert_eq!([0x3c00, 0xc000, 0x0001].map(f16), [1.0, -2.0, f32::powi(2.0, -24)]);
    }
}
