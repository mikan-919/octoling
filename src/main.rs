use octoling::gpu::Gpu;
use octoling::model::{Model, State};
use octoling::st::SafeTensors;
use std::time::Instant;

// 使い方: octoling <safetensors> <トークンID,トークンID,...> [生成するトークン数]
// 既定: GPU でプロンプトを prefill し、続けて貪欲法で生成する。
// OCTO_STEP=1: プロンプトを1トークンずつ処理し、各トークンの top-5 を出す。OCTO_CPU=1 なら CPU で同じことをする。
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = Model::load(SafeTensors::open(&args[1]).expect("重みを開けない"));
    let tokens: Vec<u32> = args[2].split(',').map(|t| t.trim().parse().unwrap()).collect();
    let n_gen: usize = args.get(3).map_or(0, |n| n.parse().unwrap());

    if std::env::var_os("OCTO_CPU").is_some() {
        let mut st = State::new();
        step_all(&tokens, |t| model.forward(&mut st, t));
        return;
    }
    let mut gpu = Gpu::new(model);
    if std::env::var_os("OCTO_STEP").is_some() {
        step_all(&tokens, |t| gpu.forward(t));
        return;
    }

    let t0 = Instant::now();
    let logits = gpu.prefill(&tokens);
    let dt = t0.elapsed().as_secs_f64();
    println!("prefill {} tokens: {:.0} ms, {:.1} tok/s", tokens.len(), dt * 1e3, tokens.len() as f64 / dt);
    report(*tokens.last().unwrap(), &logits, dt * 1e3);

    let mut next = argmax(&logits);
    let mut out = vec![];
    let t0 = Instant::now();
    for _ in 0..n_gen {
        out.push(next);
        next = argmax(&gpu.forward(next));
    }
    if n_gen > 0 {
        let dt = t0.elapsed().as_secs_f64();
        let ids: Vec<String> = out.iter().map(|t| t.to_string()).collect();
        println!("generated {n_gen} tokens: {:.1} tok/s", n_gen as f64 / dt);
        println!("{}", ids.join(","));
    }
}

fn step_all(tokens: &[u32], mut step: impl FnMut(u32) -> Vec<f32>) {
    for &t in tokens {
        let t0 = Instant::now();
        let logits = step(t);
        report(t, &logits, t0.elapsed().as_secs_f64() * 1e3);
    }
}

fn argmax(logits: &[f32]) -> u32 {
    (0..logits.len()).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).unwrap() as u32
}

/// top-5 と、logits の先頭3つと和を出す
fn report(t: u32, logits: &[f32], ms: f64) {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let top: Vec<String> = idx[..5].iter().map(|&i| format!("{i}:{:.3}", logits[i])).collect();
    println!("tok {t:>6} -> {} ({ms:.0} ms, first [{:.4} {:.4} {:.4}] sum {:.4})", top.join(" "), logits[0], logits[1], logits[2], logits.iter().sum::<f32>());
}
