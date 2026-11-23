//! serve と同じ形のプロンプトのトークン ID を出す。使い方: prompt_ids <tokenizer.json> <reasoning_effort> <本文>
use tokenizers::Tokenizer;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let t = Tokenizer::from_file(&a[1]).unwrap();
    let id = |s: &str| t.token_to_id(s).unwrap();
    let enc = |s: &str| t.encode_fast(s, false).unwrap().get_ids().to_vec();
    let mut o = vec![id("<|message_system|>"), id("<|content_text|>")];
    o.extend(enc(&format!("Thinking effort level: {}", a[2])));
    o.extend([id("<|end_message|>"), id("<|message_user|>"), id("<|content_text|>")]);
    o.extend(enc(&a[3]));
    o.push(id("<|end_message|>"));
    println!("{}", o.iter().map(u32::to_string).collect::<Vec<_>>().join(","));
}
