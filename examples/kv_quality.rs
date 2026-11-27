//! KV キャッシュの形式 (OCTO_KV) による出力の違いを測るため、長い入力の後ろの位置の logits を書き出す。
//! 使い方: kv_quality <重みのディレクトリ> <@トークン ID のファイル> <prefill するトークン数> <1 つずつ入れるトークン数> <出力ファイル>
//! 先頭を prefill し、続くトークンを 1 つずつ入れて (教師強制)、各位置の logits (f32 x 200058) を出力ファイルに続けて書く。
//! 2 つの形式の出力を examples/kv_compare.py で比べる
use octoling::gpu::Gpu;
use std::io::Write;
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let ids: Vec<u32> = std::fs::read_to_string(a[2].strip_prefix('@').unwrap()).unwrap().trim().split(',').map(|t| t.trim().parse().unwrap()).collect();
    let (n0, n1): (usize, usize) = (a[3].parse().unwrap(), a[4].parse().unwrap());
    assert!(ids.len() >= n0 + n1, "トークンが {} 個しかない", ids.len());
    let mut g = Gpu::new(octoling::model::Model::open(Path::new(&a[1])), n0 + n1);
    let mut out = std::io::BufWriter::new(std::fs::File::create(&a[5]).unwrap());
    let t0 = std::time::Instant::now();
    let mut logits = g.prefill(&ids[..n0]);
    eprintln!("prefill {n0} トークン {:.1} s", t0.elapsed().as_secs_f64());
    for &t in &ids[n0..n0 + n1] {
        out.write_all(&logits.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        logits = g.forward(t);
    }
}
