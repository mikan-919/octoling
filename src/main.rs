use octoling::gpu::Gpu;
use octoling::model::{Model, State};
use octoling::serve::{Engine, Msg, Params, Piece, effort, serve};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

const USAGE: &str = "使い方:
  octoling serve <重みのディレクトリ> [待ち受けアドレス (既定 127.0.0.1:8080)]
      OpenAI 互換の API (POST /v1/chat/completions, GET /v1/models)
  octoling chat <重みのディレクトリ> <メッセージ> [reasoning_effort (既定 high)]
      1回だけ質問し、答えを流して表示する。思考は灰色で出す
  octoling bench <重みのディレクトリ> <トークンID,トークンID,...> [生成するトークン数]
      prefill と貪欲法の生成の速さを測る。@ファイル名 でファイルから読む。; で区切ると複数のプロンプトを続けて測る (読み込みは1回、OCTO_REPEAT 回繰り返す)。OCTO_STEP=1 で1トークンずつ、OCTO_CPU=1 で CPU だけで処理する
重みのディレクトリには tokenizer.json と、公式の model-00001-of-00001.safetensors か Strata の SQ-10 (.strata) を置く
文脈長の上限は OCTO_CTX (既定 8192)。それより長くすると global 層の KV を主記憶に置く (1 トークン 32 KB)";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("{USAGE}");
        std::process::exit(2);
    }
    let dir = Path::new(&args[2]);
    let model = Model::open(dir);
    match args[1].as_str() {
        "serve" => {
            let mut engine = Engine::new(Gpu::new(model, n_ctx()), dir);
            serve(&mut engine, args.get(3).map_or("127.0.0.1:8080", String::as_str)).expect("待ち受けできない");
        }
        "chat" if args.len() >= 4 => {
            let mut engine = Engine::new(Gpu::new(model, n_ctx()), dir);
            let e = effort(&serde_json::json!(args.get(4).map_or("high", String::as_str))).unwrap_or_else(|e| panic!("{e}"));
            let msgs = [Msg { role: "user".into(), content: vec![args[3].clone()], ..Default::default() }];
            let prompt = engine.prompt(&msgs, &[], e).unwrap();
            let seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64;
            let p = Params { max_tokens: usize::MAX, temperature: 1.0, top_p: 1.0, top_k: 0, seed };
            let t0 = Instant::now();
            let (finish, n) = engine
                .generate(&prompt, &p, |piece| {
                    match piece {
                        Piece::Thinking(t) => print!("\x1b[90m{t}\x1b[0m"),
                        Piece::Text(t) => print!("{t}"),
                        Piece::Tool(_) => {}
                    }
                    std::io::stdout().flush().is_ok()
                })
                .unwrap_or_else(|e| panic!("{e}"));
            let dt = t0.elapsed().as_secs_f64();
            eprintln!("\n[{finish}: プロンプト {} トークン、生成 {n} トークン、{dt:.1} 秒、{:.1} tok/s]", prompt.len(), n as f64 / dt);
        }
        "bench" if args.len() >= 4 => bench(model, &args[3], args.get(4).map_or(0, |n| n.parse().unwrap())),
        _ => {
            eprintln!("{USAGE}");
            std::process::exit(2);
        }
    }
}

/// prompts は ; で区切った複数のプロンプト (それぞれトークン ID の列か @ファイル名)。読み込みは1回で、OCTO_REPEAT 回 (既定 1) 繰り返して測る
fn bench(model: Model, prompts: &str, n_gen: usize) {
    let prompts: Vec<Vec<u32>> = prompts
        .split(';')
        .map(|p| {
            let ids = match p.strip_prefix('@') {
                Some(path) => std::fs::read_to_string(path).expect("トークン ID のファイルを読めない"),
                None => p.to_string(),
            };
            ids.trim().split(',').map(|t| t.trim().parse().unwrap()).collect()
        })
        .collect();
    let tokens = &prompts[0];
    if std::env::var_os("OCTO_CPU").is_some() {
        let mut st = State::new();
        step_all(tokens, |t| model.forward(&mut st, t));
        return;
    }
    let mut gpu = Gpu::new(model, n_ctx());
    if std::env::var_os("OCTO_STEP").is_some() {
        step_all(tokens, |t| gpu.forward(t));
        return;
    }
    let repeat = std::env::var("OCTO_REPEAT").map_or(1, |s| s.parse().expect("OCTO_REPEAT は数"));
    for _ in 0..repeat {
        for tokens in &prompts {
            gpu.reset();
            bench_one(&mut gpu, tokens, n_gen);
        }
    }
}

fn bench_one(gpu: &mut Gpu, tokens: &[u32], n_gen: usize) {
    let t0 = Instant::now();
    let logits = gpu.prefill(tokens);
    let dt = t0.elapsed().as_secs_f64();
    println!("prefill {} tokens: {:.0} ms, {:.1} tok/s", tokens.len(), dt * 1e3, tokens.len() as f64 / dt);
    report(*tokens.last().unwrap(), &logits, dt * 1e3);

    let mut next = argmax(&logits);
    let mut out = vec![];
    let t0 = Instant::now();
    for _ in 0..n_gen {
        out.push(next);
        next = gpu.forward_greedy(next);
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

/// 文脈長の上限。OCTO_CTX で変えられる。VRAM_CTX (8192) より長いと、global 層の KV を主記憶に置く
fn n_ctx() -> usize {
    std::env::var("OCTO_CTX").ok().map_or(octoling::gpu::VRAM_CTX, |s| s.parse().expect("OCTO_CTX は数"))
}
