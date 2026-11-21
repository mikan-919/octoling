use octoling::gpu::Gpu;
use octoling::model::{Model, State};
use octoling::st::SafeTensors;
use std::time::Instant;

// 使い方: octoling <safetensors> <トークンID,トークンID,...>
// 各トークンの top-5 と、logits の先頭3つと和を出す。既定は GPU、OCTO_CPU=1 で CPU だけで計算する。
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = Model::load(SafeTensors::open(&args[1]).expect("重みを開けない"));
    let tokens: Vec<u32> = args[2].split(',').map(|t| t.trim().parse().unwrap()).collect();
    let mut step: Box<dyn FnMut(u32) -> Vec<f32>> = if std::env::var_os("OCTO_CPU").is_some() {
        let mut st = State::new();
        Box::new(move |t| model.forward(&mut st, t))
    } else {
        let mut gpu = Gpu::new(model);
        Box::new(move |t| gpu.forward(t))
    };
    for &t in &tokens {
        let t0 = Instant::now();
        let logits = step(t);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        let top: Vec<String> = idx[..5].iter().map(|&i| format!("{i}:{:.3}", logits[i])).collect();
        println!("tok {t:>6} -> {} ({:.0} ms, first [{:.4} {:.4} {:.4}] sum {:.4})", top.join(" "), ms, logits[0], logits[1], logits[2], logits.iter().sum::<f32>());
    }
}
