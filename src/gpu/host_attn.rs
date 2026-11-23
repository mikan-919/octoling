//! (生成、長い文脈の global 層) 主記憶の KV に対する注意を CPU で計算する。
//! 1トークンごとに KV 全体を読むので、PCIe (約 14 GB/s) より帯域の広い主記憶 (約 45 GB/s) の側で計算する。

use crate::model::{EXTENT, HD, N_HEAD, N_KV, N_REL};
use crate::pool;
use std::arch::x86_64::*;

const ROW: usize = N_KV * HD;
const BLK: usize = 256;
const PART: usize = HD + 2; // (最大値, 指数和, 重み付き和 [HD])
/// softmax の重みが最大の e^-30 未満のキーは V を読まない。2^18 個捨てても和の相対誤差は f32 の丸めより小さい
const PRUNE: f32 = 30.0;

/// 位置 0..n の KV に対する注意。q [24][128]、r [24][16]、sinks [24]、rel_proj [16][1024]。
/// out [24][128] は softmax の重み付き和 (sink の値はゼロベクトル)。出力ゲートはかけない
///
/// # Safety
/// kc, vc は n * 1024 個の f16 を読める
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn attend(q: &[f32], r: &[f32], sinks: &[f32], kc: *const u16, vc: *const u16, n: usize, rel_proj: &[f32], out: &mut [f32]) {
    assert!(q.len() == N_HEAD * HD && r.len() == N_HEAD * N_REL && sinks.len() == N_HEAD && rel_proj.len() == N_REL * EXTENT);
    assert!(out.len() == N_HEAD * HD && n > 0);
    let dn = EXTENT.min(n);
    let mut bias = vec![0f32; N_HEAD * dn];
    for (h, b) in bias.chunks_mut(dn).enumerate() {
        for i in 0..N_REL {
            let w = r[h * N_REL + i];
            b.iter_mut().zip(&rel_proj[i * EXTENT..][..dn]).for_each(|(b, p)| *b += w * p);
        }
    }
    let range = (n / 64).max(128);
    let mut parts = vec![0f32; n.div_ceil(range) * N_HEAD * PART];
    let (k, v) = (kc as usize, vc as usize);
    pool::chunks_mut(&mut parts, N_HEAD * PART, |c, o| {
        // SAFETY: 範囲は 0..n の中で、呼び出し側が読めることを保証している
        unsafe { part(q, &bias, dn, k as *const u16, v as *const u16, c * range, ((c + 1) * range).min(n), n - 1, o) }
    });
    for h in 0..N_HEAD {
        let sink = sinks[h];
        let m = parts.chunks(N_HEAD * PART).map(|p| p[h * PART]).fold(sink, f32::max);
        let mut z = (sink - m).exp();
        let dst = &mut out[h * HD..][..HD];
        dst.fill(0.0);
        for p in parts.chunks(N_HEAD * PART) {
            let a = &p[h * PART..][..PART];
            let f = (a[0] - m).exp();
            z += a[1] * f;
            dst.iter_mut().zip(&a[2..]).for_each(|(d, x)| *d += x * f);
        }
        dst.iter_mut().for_each(|x| *x /= z);
    }
}

/// 位置 lo..hi の分の (最大値, 指数和, 重み付き和) を o [24][PART] に書く (o は 0 で渡す)。GQA の 3 ヘッドで K, V を一度だけ読む
#[allow(clippy::too_many_arguments)]
unsafe fn part(q: &[f32], bias: &[f32], dn: usize, kc: *const u16, vc: *const u16, lo: usize, hi: usize, pos: usize, o: &mut [f32]) {
    let scale = 1.0 / (HD as f32).sqrt();
    let mut sc = [[0f32; BLK]; N_HEAD];
    for h in 0..N_HEAD {
        o[h * PART] = f32::NEG_INFINITY;
    }
    // SAFETY: q は [24][128]、o は [24][PART]。kc, vc は呼び出し側が保証する。読み書きはどれも非整列で境界内
    unsafe {
        for s0 in (lo..hi).step_by(BLK) {
            let nr = BLK.min(hi - s0);
            for r in 0..nr {
                let row = kc.add((s0 + r) * ROW);
                for kvh in 0..N_KV {
                    let qh = q.as_ptr().add(kvh * 3 * HD);
                    let (mut a0, mut a1, mut a2) = (_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps());
                    for e in (0..HD).step_by(16) {
                        let k = _mm512_cvtph_ps(_mm256_loadu_si256(row.add(kvh * HD + e).cast()));
                        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(qh.add(e)), k, a0);
                        a1 = _mm512_fmadd_ps(_mm512_loadu_ps(qh.add(HD + e)), k, a1);
                        a2 = _mm512_fmadd_ps(_mm512_loadu_ps(qh.add(2 * HD + e)), k, a2);
                    }
                    sc[kvh * 3][r] = _mm512_reduce_add_ps(a0) * scale;
                    sc[kvh * 3 + 1][r] = _mm512_reduce_add_ps(a1) * scale;
                    sc[kvh * 3 + 2][r] = _mm512_reduce_add_ps(a2) * scale;
                }
                let d = pos - (s0 + r);
                if d < dn {
                    for (h, s) in sc.iter_mut().enumerate() {
                        s[r] += bias[h * dn + d];
                    }
                }
            }
            for kvh in 0..N_KV {
                for h in kvh * 3..kvh * 3 + 3 {
                    let bm = sc[h][..nr].iter().fold(f32::NEG_INFINITY, |m, &x| if x > m { x } else { m });
                    let a = &mut o[h * PART..][..PART];
                    let (m0, mn) = (a[0], a[0].max(bm));
                    if m0 != f32::NEG_INFINITY && mn > m0 {
                        let f = (m0 - mn).exp();
                        a[1..].iter_mut().for_each(|x| *x *= f);
                    }
                    a[0] = mn;
                    a[1] += exp_block(&mut sc[h][..nr], mn);
                }
                let oh = o.as_mut_ptr().add(kvh * 3 * PART + 2);
                for r in 0..nr {
                    let p = [sc[kvh * 3][r], sc[kvh * 3 + 1][r], sc[kvh * 3 + 2][r]];
                    if p == [0.0; 3] {
                        continue;
                    }
                    let vr = vc.add((s0 + r) * ROW + kvh * HD);
                    let w = p.map(|x| _mm512_set1_ps(x));
                    for e in (0..HD).step_by(16) {
                        let v = _mm512_cvtph_ps(_mm256_loadu_si256(vr.add(e).cast()));
                        for (j, w) in w.iter().enumerate() {
                            let d = oh.add(j * PART + e);
                            _mm512_storeu_ps(d, _mm512_fmadd_ps(*w, v, _mm512_loadu_ps(d)));
                        }
                    }
                }
            }
        }
    }
}

/// s を exp(s - m) に置き換え、和を返す。s - m <= -PRUNE は 0 にする
fn exp_block(s: &mut [f32], m: f32) -> f32 {
    // SAFETY: マスクで s の範囲内だけを読み書きする
    unsafe {
        let (mv, floor) = (_mm512_set1_ps(m), _mm512_set1_ps(-PRUNE));
        let mut sum = _mm512_setzero_ps();
        for c in s.chunks_mut(16) {
            let mask = ((1u32 << c.len()) - 1) as __mmask16;
            let x = _mm512_sub_ps(_mm512_maskz_loadu_ps(mask, c.as_ptr()), mv);
            let live = _mm512_mask_cmp_ps_mask::<_CMP_GT_OQ>(mask, x, floor);
            let y = _mm512_maskz_mov_ps(live, exp_ps(x));
            _mm512_mask_storeu_ps(c.as_mut_ptr(), mask, y);
            sum = _mm512_add_ps(sum, y);
        }
        _mm512_reduce_add_ps(sum)
    }
}

/// exp(x)。x = k ln2 + f に分けて f を多項式で近似する (Cephes の係数)
#[inline(always)]
unsafe fn exp_ps(x: __m512) -> __m512 {
    // SAFETY: レジスタの演算だけ
    unsafe {
        let set = |v: f32| _mm512_set1_ps(v);
        let k = _mm512_roundscale_ps::<0x08>(_mm512_mul_ps(x, set(std::f32::consts::LOG2_E)));
        let f = _mm512_fnmadd_ps(k, set(-2.12194440e-4), _mm512_fnmadd_ps(k, set(0.693359375), x));
        let mut p = set(1.9875691500e-4);
        for c in [1.3981999507e-3, 8.3334519073e-3, 4.1665795894e-2, 1.6666665459e-1, 5.0000001201e-1] {
            p = _mm512_fmadd_ps(p, f, set(c));
        }
        let y = _mm512_add_ps(_mm512_fmadd_ps(_mm512_mul_ps(p, f), f, f), set(1.0));
        _mm512_scalef_ps(y, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *seed >> 33
    }

    /// 符号、指数 (0.25 〜 2 の範囲)、仮数が乱数の f16 と、その値
    fn half(seed: &mut u64) -> (u16, f32) {
        let (s, e, m) = (rng(seed) & 1, 13 + rng(seed) % 3, rng(seed) & 1023);
        let v = (1.0 + m as f32 / 1024.0) * 2f32.powi(e as i32 - 15) * if s == 1 { -1.0 } else { 1.0 };
        (((s << 15) | (e << 10) | m) as u16, v)
    }

    #[test]
    fn matches_scalar_softmax() {
        let mut seed = 7;
        let n = 2500;
        let (kh, kf): (Vec<u16>, Vec<f32>) = (0..n * ROW).map(|_| half(&mut seed)).unzip();
        let (vh, vf): (Vec<u16>, Vec<f32>) = (0..n * ROW).map(|_| half(&mut seed)).unzip();
        let mut f = |len: usize, scale: f32| -> Vec<f32> { (0..len).map(|_| half(&mut seed).1 * scale).collect() };
        let (q, r, sinks, rel) = (f(N_HEAD * HD, 0.5), f(N_HEAD * N_REL, 0.3), f(N_HEAD, 2.0), f(N_REL * EXTENT, 0.2));
        let mut out = vec![0.0; N_HEAD * HD];
        // SAFETY: kh, vh は n * ROW 個ある
        unsafe { attend(&q, &r, &sinks, kh.as_ptr(), vh.as_ptr(), n, &rel, &mut out) };
        for h in 0..N_HEAD {
            let kv = h / 3;
            let s: Vec<f64> = (0..n)
                .map(|j| {
                    let dot: f64 = (0..HD).map(|e| q[h * HD + e] as f64 * kf[j * ROW + kv * HD + e] as f64).sum();
                    let d = n - 1 - j;
                    let b: f64 = if d < EXTENT { (0..N_REL).map(|i| r[h * N_REL + i] as f64 * rel[i * EXTENT + d] as f64).sum() } else { 0.0 };
                    dot / (HD as f64).sqrt() + b
                })
                .collect();
            let m = s.iter().copied().fold(sinks[h] as f64, f64::max);
            let z = (sinks[h] as f64 - m).exp() + s.iter().map(|x| (x - m).exp()).sum::<f64>();
            for e in 0..HD {
                let want = s.iter().enumerate().map(|(j, x)| (x - m).exp() * vf[j * ROW + kv * HD + e] as f64).sum::<f64>() / z;
                let got = out[h * HD + e] as f64;
                assert!((got - want).abs() < 2e-5 * want.abs().max(0.01), "h={h} e={e}: {got} != {want}");
            }
        }
    }
}
