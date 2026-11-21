use octoling::model::{Model, State};
use octoling::st::SafeTensors;
use std::time::Instant;

// 使い方: octoling <safetensors> <トークンID,トークンID,...>
// 各トークンの top-5 と、最後のトークンの logits の和を出す。
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = Model::load(SafeTensors::open(&args[1]).expect("重みを開けない"));
    let tokens: Vec<u32> = args[2].split(',').map(|t| t.trim().parse().unwrap()).collect();
    let mut st = State::new();
    for &t in &tokens {
        let t0 = Instant::now();
        let logits = model.forward(&mut st, t);
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        let top: Vec<String> = idx[..5].iter().map(|&i| format!("{i}:{:.3}", logits[i])).collect();
        println!("tok {t:>6} -> {} ({:.0} ms, sum {:.4})", top.join(" "), t0.elapsed().as_secs_f64() * 1e3, logits.iter().sum::<f32>());
    }
}
