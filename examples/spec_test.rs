use octoling::gpu::Gpu;
fn argmax(l: &[f32]) -> u32 { let mut b = 0; for i in 0..l.len() { if l[i] > l[b] { b = i; } } b as u32 }
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let m = octoling::model::Model::load(octoling::st::SafeTensors::open(format!("{}/model-00001-of-00001.safetensors", a[1])).unwrap());
    let prompt: Vec<u32> = std::fs::read_to_string(&a[2]).unwrap().trim().split(',').map(|t| t.parse().unwrap()).collect();
    let mut g = Gpu::new(m);
    // 基準: prefill して 24 トークンを1つずつ貪欲に
    let mut next = argmax(&g.prefill(&prompt));
    let mut want: Vec<u32> = vec![];
    for _ in 0..24 { want.push(next); next = argmax(&g.forward(next)); }
    println!("want {:?}", want);
    // 検証: 正しい下書きと外れた下書きを混ぜて進める
    g.reset();
    let mut next = argmax(&g.prefill(&prompt));
    let mut got: Vec<u32> = vec![];
    let mut step = 0;
    while got.len() < 24 {
        let k = want.len().min(got.len() + 1 + step % 5); // 正しい下書きの数を変える
        let mut chunk: Vec<u32> = vec![next];
        chunk.extend(want[got.len() + 1..k].iter().copied());
        if step % 2 == 0 { chunk.push(12345); } // 外れの下書き
        chunk.truncate(8);
        let pred = g.verify(&chunk);
        let mut n = 1;
        while n < chunk.len() && pred[n - 1] == chunk[n] { n += 1; }
        g.rewind(n, chunk.len());
        got.extend(&chunk[..n]);
        next = pred[n - 1];
        step += 1;
    }
    got.truncate(24);
    println!("got  {:?}", got);
    println!("{}", if got == want { "MATCH" } else { "MISMATCH" });
}
