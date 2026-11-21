//! NVFP4: 4bit (E2M1) の値を2つずつ1バイトに詰め、16要素ごとに FP8 (E4M3) のスケールを持つ。
//! 値 = e2m1 × e4m3(scale) × amax / 2688。2688 = 6 (E2M1 の最大値) × 448 (E4M3 の最大値)。

pub const BLOCK: usize = 16;

pub const E2M1: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

/// OCP の E4M3 (fn): 指数バイアス 7、無限大なし、0x7f/0xff が NaN。
pub fn e4m3(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((b >> 3) & 0xf) as i32;
    let man = (b & 7) as f32;
    if exp == 0xf && b & 7 == 7 {
        return f32::NAN;
    }
    if exp == 0 {
        sign * man / 8.0 * 2f32.powi(-6)
    } else {
        sign * (1.0 + man / 8.0) * 2f32.powi(exp - 7)
    }
}

/// 1行を展開する。packed は n/2 バイト、scales は n/16 バイト。下位ニブルが偶数番目の要素。
pub fn dequant_row(packed: &[u8], scales: &[u8], global: f32, out: &mut [f32]) {
    assert_eq!(packed.len() * 2, out.len());
    assert_eq!(scales.len() * BLOCK, out.len());
    for (i, &b) in packed.iter().enumerate() {
        let s = e4m3(scales[2 * i / BLOCK]) * global;
        out[2 * i] = E2M1[(b & 0xf) as usize] * s;
        out[2 * i + 1] = E2M1[(b >> 4) as usize] * s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_values() {
        assert_eq!(e4m3(0x00), 0.0);
        assert_eq!(e4m3(0x38), 1.0);
        assert_eq!(e4m3(0x7e), 448.0);
        assert_eq!(e4m3(0xfe), -448.0);
        assert_eq!(e4m3(0x01), 2f32.powi(-9));
        assert!(e4m3(0x7f).is_nan());
    }

    #[test]
    fn dequant_block() {
        // 0x71: 下位 1 → 0.5、上位 7 → 6.0。スケール 0x38 = 1.0
        let mut out = [0.0; 16];
        dequant_row(&[0x71; 8], &[0x38], 2.0, &mut out);
        assert_eq!(&out[..2], &[1.0, 12.0]);
    }
}
