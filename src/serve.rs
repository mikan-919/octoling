//! チャットの組み立て、サンプリング、OpenAI 互換の HTTP API。
//! テンプレートは models/official/chat_template.jinja のうち、文章だけのメッセージを扱う部分を Rust で書いたもの。
//! ツール呼び出しと画像は扱わない。

use crate::gpu::{Gpu, MAX_CTX};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;

pub const MODEL_ID: &str = "inkling-2-mini";

pub struct Msg {
    pub role: String,
    pub content: Vec<String>,
    pub reasoning: Option<String>,
}

pub struct Params {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub seed: u64,
}

/// 生成した文章の断片。思考 (reasoning_content) と本文 (content) を分けて返す
pub enum Piece {
    Thinking(String),
    Text(String),
}

struct Special {
    user: u32,
    model: u32,
    system: u32,
    text: u32,
    thinking: u32,
    end_message: u32,
    end_sampling: u32,
}

pub struct Engine {
    gpu: Gpu,
    tok: Tokenizer,
    sp: Special,
    /// GPU の文脈に入っているトークン。次のプロンプトがこれの続きなら、続きの分だけ prefill する
    cached: Vec<u32>,
}

impl Engine {
    pub fn new(gpu: Gpu, tokenizer: &std::path::Path) -> Self {
        let tok = Tokenizer::from_file(tokenizer).expect("tokenizer.json を読めない");
        let id = |s: &str| tok.token_to_id(s).unwrap_or_else(|| panic!("{s} がない"));
        let sp = Special {
            user: id("<|message_user|>"),
            model: id("<|message_model|>"),
            system: id("<|message_system|>"),
            text: id("<|content_text|>"),
            thinking: id("<|content_thinking|>"),
            end_message: id("<|end_message|>"),
            end_sampling: id("<|content_model_end_sampling|>"),
        };
        Self { gpu, tok, sp, cached: vec![] }
    }

    fn encode(&self, s: &str) -> Vec<u32> {
        // 特殊トークンの文字列を本文に書かれても、特殊トークンにはしない
        self.tok.encode_fast(s, false).unwrap().get_ids().iter().copied().filter(|&t| t < 199998).collect()
    }

    /// メッセージ列をトークン列にする。最初の system 以外のメッセージの前に思考の量 (reasoning_effort) を書く
    pub fn prompt(&self, msgs: &[Msg], effort: f32) -> Result<Vec<u32>, String> {
        let sp = &self.sp;
        let mut out = vec![];
        let effort_msg = |out: &mut Vec<u32>| {
            let num = if effort == 0.0 { "0".to_string() } else { format!("{effort}") };
            out.extend([sp.system, sp.text]);
            out.extend(self.encode(&format!("Thinking effort level: {num}")));
            out.push(sp.end_message);
        };
        let mut emitted = false;
        for m in msgs {
            let role = match m.role.as_str() {
                "user" => sp.user,
                "assistant" => sp.model,
                "system" | "developer" => sp.system,
                r => return Err(format!("扱えない role: {r}")),
            };
            if !emitted && role != sp.system {
                effort_msg(&mut out);
                emitted = true;
            }
            let reasoning = m.reasoning.as_deref().filter(|r| !r.is_empty() && role == sp.model);
            if let Some(r) = reasoning {
                out.extend([role, sp.thinking]);
                out.extend(self.encode(r));
                out.push(sp.end_message);
            }
            let parts: &[String] = if m.content.is_empty() && reasoning.is_none() { &[String::new()] } else { &m.content };
            for p in parts {
                out.extend([role, sp.text]);
                out.extend(self.encode(p));
                out.push(sp.end_message);
            }
            if role == sp.model {
                out.push(sp.end_sampling);
            }
        }
        if !emitted {
            effort_msg(&mut out);
        }
        Ok(out)
    }

    /// プロンプトの続きを生成し、断片ができるたびに on に渡す。on が false を返したら止める。
    /// 返り値は (終わった理由 "stop" か "length", 生成したトークン数)
    pub fn generate(&mut self, prompt: &[u32], p: &Params, mut on: impl FnMut(Piece) -> bool) -> Result<(&'static str, usize), String> {
        if prompt.len() >= MAX_CTX {
            return Err(format!("プロンプトが {} トークンあり、文脈長の上限 {MAX_CTX} を超えている", prompt.len()));
        }
        let reuse = !self.cached.is_empty() && prompt.len() > self.cached.len() && prompt.starts_with(&self.cached);
        if !reuse {
            self.gpu.reset();
            self.cached.clear();
        }
        let reused = self.cached.len();
        let t0 = std::time::Instant::now();
        let mut logits = self.gpu.prefill(&prompt[reused..]);
        let t_prefill = t0.elapsed().as_secs_f64();
        self.cached = prompt.to_vec();
        let t0 = std::time::Instant::now();
        let log = |n: usize, finish: &str| {
            let dt = t0.elapsed().as_secs_f64();
            eprintln!(
                "prompt {} tokens ({reused} reused, prefill {:.0} ms), generated {n} tokens ({:.1} tok/s), {finish}",
                prompt.len(),
                t_prefill * 1e3,
                n as f64 / dt
            );
        };

        let mut rng = p.seed | 1;
        let mut mode = Mode::None;
        let mut pending: Vec<u32> = vec![];
        let mut n = 0;
        let finish = loop {
            if n == p.max_tokens || self.gpu.pos() >= MAX_CTX {
                break "length";
            }
            let t = sample(&logits, p, &mut rng);
            n += 1;
            if t == self.sp.end_sampling {
                break "stop";
            }
            // 特殊トークンで、思考と本文のどちらを書いているかが切り替わる
            if t >= 199998 {
                if !self.flush(&mut pending, mode, true, &mut on) {
                    break "stop";
                }
                mode = match t {
                    _ if t == self.sp.thinking => Mode::Thinking,
                    _ if t == self.sp.end_message || t == self.sp.model => Mode::None,
                    _ => Mode::Text, // <|content_text|> と、本文として扱うその他の内容
                };
            } else {
                pending.push(t);
                if !self.flush(&mut pending, mode, false, &mut on) {
                    break "stop";
                }
            }
            logits = self.gpu.forward(t);
            self.cached.push(t);
        };
        self.flush(&mut pending, mode, true, &mut on);
        log(n, finish);
        Ok((finish, n))
    }

    /// まだ渡していないトークンを文字列にして渡す。UTF-8 の途中で切れている間は (force でなければ) 待つ
    fn flush(&self, pending: &mut Vec<u32>, mode: Mode, force: bool, on: &mut impl FnMut(Piece) -> bool) -> bool {
        if pending.is_empty() {
            return true;
        }
        let s = self.tok.decode(pending, false).unwrap();
        if !force && s.contains('\u{FFFD}') {
            return true;
        }
        pending.clear();
        match mode {
            Mode::Thinking => on(Piece::Thinking(s)),
            Mode::Text => on(Piece::Text(s)),
            Mode::None => true,
        }
    }
}

#[derive(Clone, Copy)]
enum Mode {
    None,
    Thinking,
    Text,
}

fn next_rand(s: &mut u64) -> f32 {
    // xorshift64*
    *s ^= *s >> 12;
    *s ^= *s << 25;
    *s ^= *s >> 27;
    (s.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32 / (1u64 << 24) as f32
}

/// temperature、top_k、top_p でトークンを1つ選ぶ。temperature が 0 なら最大のもの。
/// ponytail: top_p だけ指定されたときは上位 1000 個の中で切る。それより外の確率は捨てる
fn sample(logits: &[f32], p: &Params, rng: &mut u64) -> u32 {
    let argmax = || (0..logits.len()).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).unwrap() as u32;
    if p.temperature <= 0.0 || p.top_k == 1 {
        return argmax();
    }
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    let limit = if p.top_k > 0 { p.top_k } else if p.top_p < 1.0 { 1000 } else { 0 };
    if limit > 0 && limit < idx.len() {
        idx.select_nth_unstable_by(limit, |&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
        idx.truncate(limit);
        idx.sort_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
    }
    let m = idx.iter().map(|&i| logits[i as usize]).fold(f32::NEG_INFINITY, f32::max);
    let mut w: Vec<f32> = idx.iter().map(|&i| ((logits[i as usize] - m) / p.temperature).exp()).collect();
    if p.top_p < 1.0 {
        let z: f32 = w.iter().sum();
        let mut acc = 0.0;
        let keep = w.iter().position(|&x| {
            acc += x / z;
            acc >= p.top_p
        });
        w.truncate(keep.map_or(w.len(), |k| k + 1));
    }
    let mut r = next_rand(rng) * w.iter().sum::<f32>();
    for (i, &x) in w.iter().enumerate() {
        r -= x;
        if r <= 0.0 {
            return idx[i];
        }
    }
    idx[w.len() - 1]
}

/// reasoning_effort は文字列 (none, minimal, low, medium, high, xhigh, max) か 0〜0.99 の数。既定は 0.9
pub fn effort(v: &Value) -> Result<f32, String> {
    let n = match v {
        Value::Null => 0.9,
        Value::String(s) => match s.trim() {
            "none" => 0.0,
            "minimal" => 0.1,
            "low" => 0.2,
            "medium" => 0.7,
            "high" => 0.9,
            "xhigh" | "max" => 0.99,
            s => return Err(format!("reasoning_effort {s} は使えない")),
        },
        v => v.as_f64().ok_or("reasoning_effort は文字列か数")? as f32,
    };
    if (0.0..=0.99).contains(&n) { Ok(n) } else { Err("reasoning_effort は 0〜0.99".into()) }
}

fn parse_messages(v: &Value) -> Result<Vec<Msg>, String> {
    let arr = v.as_array().ok_or("messages は配列")?;
    arr.iter()
        .map(|m| {
            let role = m["role"].as_str().ok_or("role がない")?.to_string();
            let content = match &m["content"] {
                Value::Null => vec![],
                Value::String(s) => vec![s.clone()],
                Value::Array(parts) => parts
                    .iter()
                    .map(|p| match p {
                        Value::String(s) => Ok(s.clone()),
                        p if matches!(p["type"].as_str(), None | Some("text" | "input_text")) => Ok(p["text"].as_str().unwrap_or("").to_string()),
                        p => Err(format!("扱えない内容: {}", p["type"])),
                    })
                    .collect::<Result<_, _>>()?,
                _ => return Err("content は文字列か配列".into()),
            };
            let reasoning = m["reasoning_content"].as_str().map(str::to_string);
            Ok(Msg { role, content, reasoning })
        })
        .collect()
}

/// OpenAI 互換の HTTP サーバー。GPU は1つの文脈しか持たないので、リクエストは1つずつ順に処理する
pub fn serve(engine: &mut Engine, addr: &str) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr)?;
    eprintln!("http://{addr}/v1 で待ち受け中 (POST /v1/chat/completions, GET /v1/models)");
    for conn in listener.incoming() {
        let mut s = conn?;
        if let Err(e) = handle(engine, &mut s) {
            eprintln!("接続のエラー: {e}");
        }
    }
    Ok(())
}

const MAX_BODY: usize = 8 << 20;

fn handle(engine: &mut Engine, s: &mut TcpStream) -> std::io::Result<()> {
    let mut r = BufReader::new(s.try_clone()?);
    let mut line = String::new();
    r.read_line(&mut line)?;
    let mut it = line.split_whitespace();
    let (method, path) = (it.next().unwrap_or("").to_string(), it.next().unwrap_or("").to_string());
    let mut len = 0;
    loop {
        let mut h = String::new();
        if r.read_line(&mut h)? == 0 || h.trim().is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':')
            && k.eq_ignore_ascii_case("content-length")
        {
            len = v.trim().parse().unwrap_or(0);
        }
    }
    if len > MAX_BODY {
        return respond(s, 413, &err_json("本文が大きすぎる"));
    }
    let mut body = vec![0; len];
    r.read_exact(&mut body)?;
    match (method.as_str(), path.trim_end_matches('/')) {
        ("GET", "/v1/models") => respond(s, 200, &json!({"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "octoling"}]})),
        ("POST", "/v1/chat/completions") => match serde_json::from_slice::<Value>(&body) {
            Ok(req) => chat(engine, &req, s),
            Err(e) => respond(s, 400, &err_json(&format!("JSON を読めない: {e}"))),
        },
        _ => respond(s, 404, &err_json(&format!("{method} {path} はない"))),
    }
}

fn err_json(msg: &str) -> Value {
    json!({"error": {"message": msg, "type": "invalid_request_error"}})
}

fn respond(s: &mut TcpStream, status: u16, v: &Value) -> std::io::Result<()> {
    let body = v.to_string();
    write!(s, "HTTP/1.1 {status} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", reason(status), body.len())
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        413 => "Payload Too Large",
        _ => "Error",
    }
}

fn chat(engine: &mut Engine, req: &Value, s: &mut TcpStream) -> std::io::Result<()> {
    if req.get("tools").is_some_and(|t| !t.is_null()) {
        return respond(s, 400, &err_json("ツール呼び出しには対応していない"));
    }
    let prompt = match parse_messages(&req["messages"]).and_then(|m| Ok((m, effort(&req["reasoning_effort"])?))).and_then(|(m, e)| engine.prompt(&m, e)) {
        Ok(p) => p,
        Err(e) => return respond(s, 400, &err_json(&e)),
    };
    let num = |k: &str| req[k].as_f64();
    let params = Params {
        max_tokens: num("max_completion_tokens").or(num("max_tokens")).map_or(MAX_CTX, |n| n as usize),
        temperature: num("temperature").unwrap_or(1.0) as f32,
        top_p: num("top_p").unwrap_or(1.0) as f32,
        top_k: num("top_k").unwrap_or(0.0) as usize,
        seed: req["seed"].as_u64().unwrap_or_else(|| SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64),
    };
    let created = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    let id = format!("chatcmpl-{created}{:x}", params.seed & 0xffffff);
    let n_prompt = prompt.len();
    let usage = |n: usize| json!({"prompt_tokens": n_prompt, "completion_tokens": n, "total_tokens": n_prompt + n});

    if req["stream"].as_bool() == Some(true) {
        write!(s, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")?;
        let chunk = |delta: Value, finish: Value| json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});
        let send = |s: &mut TcpStream, v: &Value| -> bool { write!(s, "data: {v}\n\n").and_then(|_| s.flush()).is_ok() };
        send(s, &chunk(json!({"role": "assistant", "content": ""}), Value::Null));
        let mut out = s.try_clone()?;
        let result = engine.generate(&prompt, &params, |p| {
            let delta = match p {
                Piece::Thinking(t) => json!({"reasoning_content": t}),
                Piece::Text(t) => json!({"content": t}),
            };
            send(&mut out, &chunk(delta, Value::Null)) // 接続が切れたら生成をやめる
        });
        match result {
            Ok((finish, n)) => {
                send(s, &chunk(json!({}), json!(finish)));
                if req["stream_options"]["include_usage"].as_bool() == Some(true) {
                    send(s, &json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID, "choices": [], "usage": usage(n)}));
                }
            }
            Err(e) => {
                send(s, &err_json(&e));
            }
        }
        write!(s, "data: [DONE]\n\n")?;
        return s.flush();
    }

    let (mut thinking, mut text) = (String::new(), String::new());
    let result = engine.generate(&prompt, &params, |p| {
        match p {
            Piece::Thinking(t) => thinking += &t,
            Piece::Text(t) => text += &t,
        }
        true
    });
    match result {
        Ok((finish, n)) => {
            let mut msg = json!({"role": "assistant", "content": text});
            if !thinking.is_empty() {
                msg["reasoning_content"] = json!(thinking);
            }
            respond(s, 200, &json!({"id": id, "object": "chat.completion", "created": created, "model": MODEL_ID,
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage(n)}))
        }
        Err(e) => respond(s, 400, &err_json(&e)),
    }
}
