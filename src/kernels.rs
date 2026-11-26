//! AVX-512 の内積。x は f32 のまま、重みだけを展開して積和する。

use crate::nvfp4::{E2M1, E4M3};
use std::arch::x86_64::*;

#[cfg(not(target_feature = "avx512f"))]
compile_error!("AVX-512 が必要。.cargo/config.toml の target-cpu=native で有効になる");

/// BF16 の行と x の内積。n は 32 の倍数。
pub fn dot_bf16(w: &[u16], x: &[f32]) -> f32 {
    assert!(w.len() == x.len() && x.len().is_multiple_of(32));
    // SAFETY: 長さは上で検査した。読み出しは境界内の非整列ロードだけ
    unsafe {
        let (mut a0, mut a1) = (_mm512_setzero_ps(), _mm512_setzero_ps());
        for i in (0..x.len()).step_by(32) {
            let w0 = _mm512_cvtepu16_epi32(_mm256_loadu_si256(w.as_ptr().add(i).cast()));
            let w1 = _mm512_cvtepu16_epi32(_mm256_loadu_si256(w.as_ptr().add(i + 16).cast()));
            a0 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(w0, 16)), _mm512_loadu_ps(x.as_ptr().add(i)), a0);
            a1 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(w1, 16)), _mm512_loadu_ps(x.as_ptr().add(i + 16)), a1);
        }
        _mm512_reduce_add_ps(_mm512_add_ps(a0, a1))
    }
}

/// NVFP4 の行と x の内積。amax による大域スケールは掛けない。
pub fn dot_nvfp4(packed: &[u8], scales: &[u8], x: &[f32]) -> f32 {
    assert!(packed.len() * 2 == x.len() && scales.len() * 16 == x.len());
    let st = &*E4M3;
    // SAFETY: 長さは上で検査した。読み出しは境界内の非整列ロードだけ
    unsafe {
        let lut = _mm512_loadu_ps(E2M1.as_ptr());
        let mask = _mm_set1_epi8(0x0f);
        let mut acc = _mm512_setzero_ps();
        for (blk, &s) in scales.iter().enumerate() {
            let v = _mm_loadl_epi64(packed.as_ptr().add(blk * 8).cast());
            // 下位ニブルが偶数番目、上位ニブルが奇数番目の要素
            let nib = _mm_unpacklo_epi8(_mm_and_si128(v, mask), _mm_and_si128(_mm_srli_epi16(v, 4), mask));
            let w = _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(nib), lut);
            let xs = _mm512_mul_ps(_mm512_loadu_ps(x.as_ptr().add(blk * 16)), _mm512_set1_ps(st[s as usize]));
            acc = _mm512_fmadd_ps(w, xs, acc);
        }
        _mm512_reduce_add_ps(acc)
    }
}

/// MXFP8 の行と x の内積。E4M3 のビットを f32 の指数部と仮数部へずらすと 2^-120 倍の値になるので、最後に 2^120 を掛ける。
/// 0x7f/0xff (NaN) は扱わない。変換スクリプトがチェックポイントに NaN がないことを確かめている。
pub fn dot_mxfp8(w: &[u8], scales: &[u8], x: &[f32]) -> f32 {
    assert!(w.len() == x.len() && scales.len() * 32 == x.len());
    // SAFETY: 長さは上で検査した。読み出しは境界内の非整列ロードだけ
    unsafe {
        let m7f = _mm512_set1_epi32(0x7f);
        let m80 = _mm512_set1_epi32(0x80);
        let fp8 = |p: *const u8| {
            let b = _mm512_cvtepu8_epi32(_mm_loadu_si128(p.cast()));
            let mag = _mm512_slli_epi32(_mm512_and_si512(b, m7f), 20);
            let sign = _mm512_slli_epi32(_mm512_and_si512(b, m80), 24);
            _mm512_castsi512_ps(_mm512_or_si512(mag, sign))
        };
        let mut acc = _mm512_setzero_ps();
        for (blk, &s) in scales.iter().enumerate() {
            let i = blk * 32;
            let b = _mm512_mul_ps(fp8(w.as_ptr().add(i)), _mm512_loadu_ps(x.as_ptr().add(i)));
            let b = _mm512_fmadd_ps(fp8(w.as_ptr().add(i + 16)), _mm512_loadu_ps(x.as_ptr().add(i + 16)), b);
            // 2^(s-127)。s=0 (2^-127) は 0 になるが、影響は無視できる
            acc = _mm512_fmadd_ps(b, _mm512_set1_ps(f32::from_bits((s as u32) << 23)), acc);
        }
        _mm512_reduce_add_ps(acc) * 2f32.powi(120)
    }
}

/// f32 どうしの内積。n は 32 の倍数。
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert!(a.len() == b.len() && a.len().is_multiple_of(32));
    // SAFETY: 長さは上で検査した
    unsafe {
        let (mut a0, mut a1) = (_mm512_setzero_ps(), _mm512_setzero_ps());
        for i in (0..a.len()).step_by(32) {
            a0 = _mm512_fmadd_ps(_mm512_loadu_ps(a.as_ptr().add(i)), _mm512_loadu_ps(b.as_ptr().add(i)), a0);
            a1 = _mm512_fmadd_ps(_mm512_loadu_ps(a.as_ptr().add(i + 16)), _mm512_loadu_ps(b.as_ptr().add(i + 16)), a1);
        }
        _mm512_reduce_add_ps(_mm512_add_ps(a0, a1))
    }
}

/// 2本の w と4本の x の、8通りの内積。読み込み6回で FMA 8回を回し、8本の独立な積和で FMA の待ち時間を埋める。
pub fn dot_2x4(w: [&[f32]; 2], x: [&[f32]; 4]) -> [[f32; 4]; 2] {
    let n = w[0].len();
    assert!(n.is_multiple_of(16) && w[1].len() == n && x.iter().all(|x| x.len() == n));
    // SAFETY: 長さは上で検査した
    unsafe {
        let mut acc = [[_mm512_setzero_ps(); 4]; 2];
        for i in (0..n).step_by(16) {
            let w0 = _mm512_loadu_ps(w[0].as_ptr().add(i));
            let w1 = _mm512_loadu_ps(w[1].as_ptr().add(i));
            for j in 0..4 {
                let xv = _mm512_loadu_ps(x[j].as_ptr().add(i));
                acc[0][j] = _mm512_fmadd_ps(w0, xv, acc[0][j]);
                acc[1][j] = _mm512_fmadd_ps(w1, xv, acc[1][j]);
            }
        }
        acc.map(|r| r.map(|a| _mm512_reduce_add_ps(a)))
    }
}

/// MXFP8 の1行を f32 に展開する。
pub fn dequant_mxfp8(w: &[u8], scales: &[u8], out: &mut [f32]) {
    assert!(w.len() == out.len() && scales.len() * 32 == out.len());
    // SAFETY: 長さは上で検査した。dot_mxfp8 と同じビットの置き換え
    unsafe {
        let m7f = _mm512_set1_epi32(0x7f);
        let m80 = _mm512_set1_epi32(0x80);
        for (blk, &s) in scales.iter().enumerate() {
            // 2^120 を先に掛けて E4M3 の値に戻してから、2^(s-127) を掛ける (逆順だと s が大きいとき溢れる)
            let sc = _mm512_set1_ps(f32::from_bits((s as u32) << 23));
            let k = _mm512_set1_ps(2f32.powi(120));
            for half in 0..2 {
                let i = blk * 32 + half * 16;
                let b = _mm512_cvtepu8_epi32(_mm_loadu_si128(w.as_ptr().add(i).cast()));
                let mag = _mm512_slli_epi32(_mm512_and_si512(b, m7f), 20);
                let sign = _mm512_slli_epi32(_mm512_and_si512(b, m80), 24);
                let v = _mm512_mul_ps(_mm512_castsi512_ps(_mm512_or_si512(mag, sign)), k);
                _mm512_storeu_ps(out.as_mut_ptr().add(i), _mm512_mul_ps(v, sc));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nvfp4::{dequant_row, e4m3};

    fn rand_bytes(n: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (s >> 33) as u8
            })
            .collect()
    }

    fn rand_x(n: usize) -> Vec<f32> {
        rand_bytes(n, 7).iter().map(|&b| b as f32 / 128.0 - 1.0).collect()
    }

    fn close(a: f32, b: f32) {
        assert!((a - b).abs() <= 1e-4 * b.abs().max(1.0), "{a} != {b}");
    }

    #[test]
    fn bf16_matches_scalar() {
        let x = rand_x(2304);
        let w: Vec<u16> = x.iter().map(|v| ((v * 0.37).to_bits() >> 16) as u16).collect();
        let want: f32 = w.iter().zip(&x).map(|(&w, x)| f32::from_bits((w as u32) << 16) * x).sum();
        close(dot_bf16(&w, &x), want);
    }

    #[test]
    fn nvfp4_matches_dequant() {
        let x = rand_x(2304);
        let packed = rand_bytes(1152, 1);
        // 0x7f/0xff (NaN) を避けたスケール
        let scales: Vec<u8> = rand_bytes(144, 2).iter().map(|b| b & 0x77).collect();
        let mut row = vec![0.0; 2304];
        dequant_row(&packed, &scales, 1.0, &mut row);
        close(dot_nvfp4(&packed, &scales, &x), row.iter().zip(&x).map(|(w, x)| w * x).sum());
    }

    #[test]
    fn mxfp8_matches_scalar() {
        let x = rand_x(1152);
        let w: Vec<u8> = rand_bytes(1152, 3).iter().map(|&b| if b & 0x7f == 0x7f { 0 } else { b }).collect();
        let scales: Vec<u8> = rand_bytes(36, 4).iter().map(|b| 110 + b % 30).collect();
        let want: f32 = (0..1152).map(|i| e4m3(w[i]) * 2f32.powi(scales[i / 32] as i32 - 127) * x[i]).sum();
        close(dot_mxfp8(&w, &scales, &x), want);
        let mut row = vec![0.0; 1152];
        dequant_mxfp8(&w, &scales, &mut row);
        close(dot_f32(&row, &x), want);
        close(dot_2x4([&x, &row], [&x; 4])[1][3], want);
    }
}
