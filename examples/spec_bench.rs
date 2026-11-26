//! 投機的デコードの速さと内訳を測り、出力が1トークンずつの貪欲法と同じかを確かめる。
//! 使い方: spec_bench <重みのディレクトリ> <プロンプト;プロンプト...> <生成するトークン数>
//! プロンプトはトークン ID の列か @ファイル名。OCTO_REPEAT 回繰り返す
//! OCTO_FORCE=ファイル (トークン ID の列) を付けると、投機的デコードの本体の選択をその列に置き換える (同じ文での下書きの当たり方と速さを比べる)
use octoling::gpu::{Drafter, Gpu, SPEC_T, VRAM_CTX};
use std::path::Path;
use std::time::Instant;

fn argmax(l: &[f32]) -> u32 {
    (0..l.len()).max_by(|&a, &b| l[a].total_cmp(&l[b]).then(b.cmp(&a))).unwrap() as u32
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dir = Path::new(&a[1]);
    let n: usize = a[3].parse().unwrap();
    let prompts: Vec<Vec<u32>> = a[2]
        .split(';')
        .map(|p| {
            let s = p.strip_prefix('@').map_or(p.to_string(), |f| std::fs::read_to_string(f).unwrap());
            s.trim().split(',').map(|t| t.trim().parse().unwrap()).collect()
        })
        .collect();
    let m = octoling::model::Model::open(dir);
    let ctx = std::env::var("OCTO_CTX").map_or(VRAM_CTX, |s| s.parse().unwrap());
    let mut g = Gpu::new(m, ctx);
    let mut d = Drafter::load(&g, &dir.join("mtp.safetensors"));
    let repeat = std::env::var("OCTO_REPEAT").map_or(1, |s| s.parse().unwrap());
    let force: Option<Vec<u32>> =
        std::env::var("OCTO_FORCE").ok().map(|f| std::fs::read_to_string(f).unwrap().trim().split(',').map(|t| t.parse().unwrap()).collect());
    for _ in 0..repeat {
        for p in &prompts {
            g.reset();
            let mut next = argmax(&g.prefill(p));
            let mut want = vec![];
            let t0 = Instant::now();
            for _ in 0..n {
                want.push(next);
                next = g.forward_greedy(next);
            }
            let base = n as f64 / t0.elapsed().as_secs_f64();

            g.reset();
            d.reset();
            let mut next = argmax(&g.prefill_with(p, |g, c| d.add_ctx(g, c)));
            if let Some(f) = &force {
                next = f[0];
            }
            let mut got: Vec<u32> = vec![];
            let (mut steps, mut sent, mut hist) = (0usize, 0usize, [0usize; SPEC_T + 1]);
            let mut ms = [0f64; 3];
            let t0 = Instant::now();
            while got.len() < n {
                let room = (n - got.len()).min(SPEC_T);
                let mut chunk = vec![next];
                let t = Instant::now();
                chunk.extend(d.draft(&g, next, room - 1));
                ms[0] += t.elapsed().as_secs_f64();
                let t = Instant::now();
                let mut pred = g.verify(&chunk);
                if let Some(f) = &force {
                    let b = got.len();
                    pred = (0..chunk.len()).map(|i| f[(b + i + 1).min(f.len() - 1)]).collect();
                }
                ms[1] += t.elapsed().as_secs_f64();
                let mut k = 1;
                while k < chunk.len() && pred[k - 1] == chunk[k] {
                    k += 1;
                }
                let t = Instant::now();
                g.rewind(k, chunk.len());
                d.add_ctx(&g, k);
                ms[2] += t.elapsed().as_secs_f64();
                got.extend(&chunk[..k]);
                next = pred[k - 1];
                steps += 1;
                sent += chunk.len();
                hist[chunk.len()] += 1;
            }
            let dt = t0.elapsed().as_secs_f64();
            got.truncate(n);
            if let Some(i) = (0..n).find(|&i| got[i] != want[i]) {
                g.reset();
                let mut l = g.prefill(p);
                for &t in &want[..i] {
                    l = g.forward(t);
                }
                let mut s = l.clone();
                s.sort_by(|a, b| b.total_cmp(a));
                println!("  first diff at {i}: greedy {} spec {}, top-2 gap {:.5}", want[i], got[i], s[0] - s[1]);
            }
            let per = 1e3 / steps as f64;
            println!(
                "{} tokens: greedy {base:.1} tok/s, spec {:.1} tok/s, {} | accepted {:.2}/step, sent {:.2}/step, draft {:.1} + verify {:.1} + ctx {:.1} ms/step, lens {:?}",
                p.len(),
                n as f64 / dt,
                if got == want { "MATCH" } else { "MISMATCH" },
                n as f64 / steps as f64,
                sent as f64 / steps as f64,
                ms[0] * per,
                ms[1] * per,
                ms[2] * per,
                &hist[1..]
            );
        }
    }
}
