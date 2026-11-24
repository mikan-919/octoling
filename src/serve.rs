//! チャットの組み立て、サンプリング、OpenAI 互換の HTTP API。
//! テンプレートは models/official/chat_template.jinja のうち、文章とツール呼び出しを扱う部分を Rust で書いたもの。
//! 画像は扱わない。

use crate::gpu::{Drafter, Gpu, KvSnap, SPEC_T};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;

pub const MODEL_ID: &str = "inkling-2-mini";

#[derive(Default)]
pub struct Msg {
    pub role: String,
    pub content: Vec<String>,
    pub reasoning: Option<String>,
    pub name: Option<String>,             // role が tool のとき、結果を返すツールの名前
    pub tool_calls: Vec<(String, Value)>, // (ツール名, 引数のオブジェクト)
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
    Tool(String), // ツール呼び出し1つの JSON ({"name": ..., "args": {...}})
}

struct Special {
    user: u32,
    model: u32,
    system: u32,
    text: u32,
    thinking: u32,
    end_message: u32,
    end_sampling: u32,
    tool: u32,
    xml: u32,
    invoke_tool: u32,
}

pub struct Engine {
    gpu: Gpu,
    tok: Tokenizer,
    sp: Special,
    /// GPU の文脈に入っているトークン。次のプロンプトがこれの続きなら、続きの分だけ prefill する
    cached: Vec<u32>,
    /// 主記憶に残したプロンプトの文脈。新しく使った順。次のプロンプトがどれかの続きなら、そこから続ける
    kv: Vec<KvSnap>,
    kv_max: usize, // kv の合計の上限 (バイト)。OCTO_KV_GB で変えられる
    draft: Option<Drafter>,
}

impl Engine {
    /// dir の tokenizer.json を読む。OCTO_DRAFT=1 で mtp.safetensors があれば下書きモデルとして読み、生成を投機的デコードにする。
    /// 公式の重みでは検証が遅く、下書きの時間を取り返せないことが多いので、既定では使わない
    pub fn new(gpu: Gpu, dir: &std::path::Path) -> Self {
        let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json を読めない");
        let mtp = dir.join("mtp.safetensors");
        let want = mtp.exists() && std::env::var_os("OCTO_DRAFT").is_some();
        if want && gpu.host_kv() {
            eprintln!("global 層の KV が主記憶にあるので下書きモデルは使わない");
        }
        let draft = (want && !gpu.host_kv()).then(|| Drafter::load(&gpu, &mtp));
        let kv_max = std::env::var("OCTO_KV_GB").ok().map_or(4.0, |s| s.parse::<f64>().expect("OCTO_KV_GB は数"));
        let id = |s: &str| tok.token_to_id(s).unwrap_or_else(|| panic!("{s} がない"));
        let sp = Special {
            user: id("<|message_user|>"),
            model: id("<|message_model|>"),
            system: id("<|message_system|>"),
            text: id("<|content_text|>"),
            thinking: id("<|content_thinking|>"),
            end_message: id("<|end_message|>"),
            end_sampling: id("<|content_model_end_sampling|>"),
            tool: id("<|message_tool|>"),
            xml: id("<|content_xml|>"),
            invoke_tool: id("<|content_invoke_tool_json|>"),
        };
        Self { gpu, tok, sp, cached: vec![], kv: vec![], kv_max: (kv_max * 1e9) as usize, draft }
    }

    fn encode(&self, s: &str) -> Vec<u32> {
        // 特殊トークンの文字列を本文に書かれても、特殊トークンにはしない
        self.tok.encode_fast(s, false).unwrap().get_ids().iter().copied().filter(|&t| t < 199998).collect()
    }

    /// メッセージ列をトークン列にする。最初の system 以外のメッセージの前に思考の量 (reasoning_effort) を書く
    pub fn prompt(&self, msgs: &[Msg], tools: &[Value], effort: f32) -> Result<Vec<u32>, String> {
        let sp = &self.sp;
        let mut out = vec![];
        if !tools.is_empty() {
            // serde_json の Map はキーの順に並ぶので、テンプレートの sort_keys と同じ文字列になる
            let specs: Vec<Value> = tools
                .iter()
                .map(|t| {
                    let f = t.get("function").unwrap_or(t);
                    let parameters = if f["parameters"].is_null() { json!({}) } else { f["parameters"].clone() };
                    json!({"description": f["description"].as_str().unwrap_or(""), "name": f["name"], "parameters": parameters, "type": t["type"].as_str().unwrap_or("function")})
                })
                .collect();
            out.push(sp.system);
            out.extend(self.encode("tool_declare"));
            out.push(sp.xml);
            out.extend(self.encode(&Value::from(specs).to_string()));
            out.push(sp.end_message);
        }
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
                "tool" => sp.tool,
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
            let skip_empty = role == sp.model && (reasoning.is_some() || !m.tool_calls.is_empty());
            let parts: &[String] = if m.content.is_empty() && !skip_empty { &[String::new()] } else { &m.content };
            let header: Vec<u32> = std::iter::once(role).chain(m.name.iter().filter(|_| role == sp.tool).flat_map(|n| self.encode(n))).collect();
            for p in parts.iter().filter(|p| !(skip_empty && p.is_empty())) {
                out.extend(&header);
                out.push(sp.text);
                out.extend(self.encode(p));
                out.push(sp.end_message);
            }
            for (name, args) in &m.tool_calls {
                out.push(sp.model);
                out.extend(self.encode(name));
                out.push(sp.invoke_tool);
                out.extend(self.encode(&format!("{{\"name\":{},\"args\":{args}}}", Value::from(name.as_str()))));
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
        let max_ctx = self.gpu.max_ctx();
        if prompt.len() >= max_ctx {
            return Err(format!("プロンプトが {} トークンあり、文脈長の上限 {max_ctx} を超えている", prompt.len()));
        }
        let live = if prompt.len() > self.cached.len() && prompt.starts_with(&self.cached) { self.cached.len() } else { 0 };
        let best = (0..self.kv.len()).filter(|&i| self.kv[i].tokens.len() < prompt.len() && prompt.starts_with(&self.kv[i].tokens)).max_by_key(|&i| self.kv[i].tokens.len());
        if let Some(i) = best.filter(|&i| self.kv[i].tokens.len() > live) {
            self.kv[..=i].rotate_right(1);
            let s = &self.kv[0];
            let keep = self.cached.iter().zip(&s.tokens).take_while(|(a, b)| a == b).count();
            self.gpu.load_kv(self.draft.as_mut(), s, keep);
            self.cached.clone_from(&s.tokens);
        } else if live == 0 {
            self.gpu.reset();
            self.cached.clear();
            if let Some(d) = &mut self.draft {
                d.reset();
            }
        }
        let reused = self.cached.len();
        let t0 = std::time::Instant::now();
        // tools と system は会話が変わっても同じことが多いので、最初の user の前までを別に残す
        let head = prompt.iter().position(|&t| t == self.sp.user).unwrap_or(0);
        if head > reused {
            self.prefill(&prompt[reused..head]);
            self.cached.extend_from_slice(&prompt[reused..head]);
            self.remember(&prompt[..head], true);
        }
        let done = self.cached.len();
        let mut logits = self.prefill(&prompt[done..]);
        let t_prefill = t0.elapsed().as_secs_f64();
        self.cached = prompt.to_vec();
        self.remember(prompt, false);
        let t0 = std::time::Instant::now();

        let mut out = Out { mode: Mode::None, pending: vec![] };
        let (mut n, mut steps, mut t_draft, mut t_verify) = (0, 0, 0f64, 0f64);
        let greedy = p.temperature <= 0.0 || p.top_k == 1;
        let mut rng = p.seed | 1;
        let finish = if self.draft.is_some() {
            // 投機的デコード: 下書きをまとめて検証し、本体の分布から選んだトークンと一致した先頭までを受け入れる。
            // 1行ごとの選び方は1トークンずつ生成するときと同じなので、出力の分布は変わらない
            let mut next = sample(&logits, p, &mut rng);
            loop {
                if n >= p.max_tokens || self.gpu.pos() >= max_ctx {
                    break "length";
                }
                if next == self.sp.end_sampling {
                    n += 1;
                    break "stop";
                }
                let room = (p.max_tokens - n).min(max_ctx - self.gpu.pos()).min(SPEC_T);
                let mut chunk = vec![next];
                let tm = std::time::Instant::now();
                chunk.extend(self.draft.as_mut().unwrap().draft(&self.gpu, next, room - 1));
                t_draft += tm.elapsed().as_secs_f64();
                let tm = std::time::Instant::now();
                let amax = self.gpu.verify(&chunk);
                let mut a = 0;
                next = loop {
                    a += 1;
                    let pick = if greedy { amax[a - 1] } else { sample(&self.gpu.verify_logits(a - 1), p, &mut rng) };
                    if a == chunk.len() || pick != chunk[a] || chunk[a - 1] == self.sp.end_sampling {
                        break pick;
                    }
                };
                t_verify += tm.elapsed().as_secs_f64();
                self.gpu.rewind(a, chunk.len());
                self.draft.as_mut().unwrap().add_ctx(&self.gpu, a);
                self.cached.extend_from_slice(&chunk[..a]);
                steps += 1;
                let mut stop = false;
                for &t in &chunk[..a] {
                    n += 1;
                    if t == self.sp.end_sampling || !self.emit(t, &mut out, &mut on) {
                        stop = true;
                        break;
                    }
                }
                if stop {
                    break "stop";
                }
            }
        } else {
            // 貪欲法なら、次のトークンは GPU が logits から選ぶ
            let mut next = greedy.then(|| argmax(&logits));
            loop {
                if n == p.max_tokens || self.gpu.pos() >= max_ctx {
                    break "length";
                }
                let t = next.take().unwrap_or_else(|| sample(&logits, p, &mut rng));
                n += 1;
                if t == self.sp.end_sampling || !self.emit(t, &mut out, &mut on) {
                    break "stop";
                }
                if greedy {
                    next = Some(self.gpu.forward_greedy(t));
                } else {
                    logits = self.gpu.forward(t);
                }
                self.cached.push(t);
            }
        };
        self.flush(&mut out.pending, out.mode, true, &mut on);
        let dt = t0.elapsed().as_secs_f64();
        let spec = if steps > 0 {
            let k = 1e3 / steps as f64;
            format!(", {:.2} tokens per verify, draft {:.1} ms + verify {:.1} ms per step", n as f64 / steps as f64, t_draft * k, t_verify * k)
        } else {
            String::new()
        };
        eprintln!(
            "prompt {} tokens ({reused} reused, prefill {:.0} ms), generated {n} tokens ({:.1} tok/s{spec}), {finish}",
            prompt.len(),
            t_prefill * 1e3,
            n as f64 / dt
        );
        Ok((finish, n))
    }

    /// 本体と下書きモデルの文脈を tokens の分だけ進め、最後の logits を返す
    fn prefill(&mut self, tokens: &[u32]) -> Vec<f32> {
        let draft = &mut self.draft;
        self.gpu.prefill_with(tokens, |g, n| {
            if let Some(d) = draft {
                d.add_ctx(g, n);
            }
        })
    }

    /// 今の文脈を主記憶に残す。tokens の先頭に当たる記録があれば global の KV を引き継ぎ、古い記録は置き換える
    fn remember(&mut self, tokens: &[u32], head: bool) {
        if self.kv_max == 0 {
            return;
        }
        let base = (0..self.kv.len()).filter(|&i| tokens.starts_with(&self.kv[i].tokens)).max_by_key(|&i| self.kv[i].tokens.len());
        let mut s = match base {
            Some(i) if self.kv[i].head => self.kv[i].clone(),
            Some(i) => self.kv.remove(i),
            None => KvSnap::default(),
        };
        s.head = head;
        self.gpu.save_kv(self.draft.as_ref(), &mut s, tokens);
        self.kv.insert(0, s);
        let mut total: usize = self.kv.iter().map(KvSnap::bytes).sum();
        while total > self.kv_max && self.kv.len() > 1 {
            total -= self.kv.pop().unwrap().bytes();
        }
    }

    /// 生成したトークン t を断片として渡す。特殊トークンで、思考と本文のどちらを書いているかが切り替わる。
    /// 受け手が止めたら false
    fn emit(&self, t: u32, out: &mut Out, on: &mut impl FnMut(Piece) -> bool) -> bool {
        if t >= 199998 {
            let ok = if matches!(out.mode, Mode::Tool) {
                let s = self.tok.decode(&out.pending, false).unwrap();
                out.pending.clear();
                on(Piece::Tool(s))
            } else {
                self.flush(&mut out.pending, out.mode, true, on)
            };
            out.mode = match t {
                _ if t == self.sp.thinking => Mode::Thinking,
                _ if t == self.sp.invoke_tool => Mode::Tool,
                _ if t == self.sp.end_message || t == self.sp.model => Mode::None,
                _ => Mode::Text, // <|content_text|> と、本文として扱うその他の内容
            };
            ok
        } else {
            out.pending.push(t);
            matches!(out.mode, Mode::Tool) || self.flush(&mut out.pending, out.mode, false, on)
        }
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
            Mode::None | Mode::Tool => true, // 途中で切れたツール呼び出しは捨てる
        }
    }
}

struct Out {
    mode: Mode,
    pending: Vec<u32>,
}

/// 最大の値の位置。同じ値なら小さい位置 (GPU の argmax_rows と同じ決め方)
fn argmax(l: &[f32]) -> u32 {
    let mut b = 0;
    for (i, &x) in l.iter().enumerate() {
        if x > l[b] {
            b = i;
        }
    }
    b as u32
}

#[derive(Clone, Copy)]
enum Mode {
    None,
    Thinking,
    Text,
    Tool,
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
    if p.temperature <= 0.0 || p.top_k == 1 {
        return argmax(logits);
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
    let mut names = HashMap::new(); // tool_call_id → ツール名
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
            let mut tool_calls = vec![];
            for tc in m["tool_calls"].as_array().into_iter().flatten() {
                let f = &tc["function"];
                let name = f["name"].as_str().ok_or("ツール呼び出しに名前がない")?.to_string();
                let args = match &f["arguments"] {
                    Value::Null => json!({}),
                    Value::String(s) if s.trim().is_empty() => json!({}),
                    Value::String(s) => serde_json::from_str(s).map_err(|e| format!("{name} の引数が JSON ではない: {e}"))?,
                    v => v.clone(),
                };
                if !args.is_object() {
                    return Err(format!("{name} の引数がオブジェクトではない"));
                }
                if let Some(id) = tc["id"].as_str() {
                    names.insert(id.to_string(), name.clone());
                }
                tool_calls.push((name, args));
            }
            let name = m["name"].as_str().map(str::to_string).or_else(|| m["tool_call_id"].as_str().and_then(|id| names.get(id).cloned()));
            Ok(Msg { role, content, reasoning, name, tool_calls })
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

/// モデルが書いたツール呼び出しの JSON を OpenAI の形にする。読めなければ None
fn tool_call(text: &str, id: String, index: usize) -> Option<Value> {
    let v: Value = serde_json::from_str(text).ok()?;
    let args = v.get("args").filter(|a| a.is_object())?;
    Some(json!({"index": index, "id": id, "type": "function", "function": {"name": v["name"].as_str()?, "arguments": args.to_string()}}))
}

fn chat(engine: &mut Engine, req: &Value, s: &mut TcpStream) -> std::io::Result<()> {
    let tools = if req["tool_choice"] == "none" { vec![] } else { req["tools"].as_array().cloned().unwrap_or_default() };
    let prompt = match parse_messages(&req["messages"]).and_then(|m| Ok((m, effort(&req["reasoning_effort"])?))).and_then(|(m, e)| engine.prompt(&m, &tools, e)) {
        Ok(p) => p,
        Err(e) => return respond(s, 400, &err_json(&e)),
    };
    let num = |k: &str| req[k].as_f64();
    let params = Params {
        max_tokens: num("max_completion_tokens").or(num("max_tokens")).map_or(engine.gpu.max_ctx(), |n| n as usize),
        temperature: num("temperature").unwrap_or(1.0) as f32,
        top_p: num("top_p").unwrap_or(1.0) as f32,
        top_k: num("top_k").unwrap_or(0.0) as usize,
        seed: req["seed"].as_u64().unwrap_or_else(|| SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64),
    };
    let created = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    let id = format!("chatcmpl-{created}{:x}", params.seed & 0xffffff);
    let call_id = |i: usize| format!("call_{created:x}{:x}_{i}", params.seed & 0xffffff);
    let n_prompt = prompt.len();
    let usage = |n: usize| json!({"prompt_tokens": n_prompt, "completion_tokens": n, "total_tokens": n_prompt + n});

    if req["stream"].as_bool() == Some(true) {
        write!(s, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")?;
        let chunk = |delta: Value, finish: Value| json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});
        let send = |s: &mut TcpStream, v: &Value| -> bool { write!(s, "data: {v}\n\n").and_then(|_| s.flush()).is_ok() };
        send(s, &chunk(json!({"role": "assistant", "content": ""}), Value::Null));
        let mut out = s.try_clone()?;
        let mut calls = 0;
        let result = engine.generate(&prompt, &params, |p| {
            let delta = match p {
                Piece::Thinking(t) => json!({"reasoning_content": t}),
                Piece::Text(t) => json!({"content": t}),
                Piece::Tool(t) => match tool_call(&t, call_id(calls), calls) {
                    Some(c) => {
                        calls += 1;
                        json!({"tool_calls": [c]})
                    }
                    None => json!({"content": t}),
                },
            };
            send(&mut out, &chunk(delta, Value::Null)) // 接続が切れたら生成をやめる
        });
        match result {
            Ok((finish, n)) => {
                let finish = if calls > 0 && finish == "stop" { "tool_calls" } else { finish };
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

    let (mut thinking, mut text, mut calls) = (String::new(), String::new(), vec![]);
    let result = engine.generate(&prompt, &params, |p| {
        match p {
            Piece::Thinking(t) => thinking += &t,
            Piece::Text(t) => text += &t,
            Piece::Tool(t) => match tool_call(&t, call_id(calls.len()), calls.len()) {
                Some(c) => calls.push(c),
                None => text += &t,
            },
        }
        true
    });
    match result {
        Ok((finish, n)) => {
            let mut msg = json!({"role": "assistant", "content": text});
            if !thinking.is_empty() {
                msg["reasoning_content"] = json!(thinking);
            }
            let finish = if calls.is_empty() || finish != "stop" { finish } else { "tool_calls" };
            if !calls.is_empty() {
                msg["tool_calls"] = json!(calls);
            }
            respond(s, 200, &json!({"id": id, "object": "chat.completion", "created": created, "model": MODEL_ID,
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage(n)}))
        }
        Err(e) => respond(s, 400, &err_json(&e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_model_tool_call() {
        let c = tool_call(r#"{"name":"read","args":{"path":"a.txt"}}"#, "x".into(), 2).unwrap();
        assert_eq!(c["index"], 2);
        assert_eq!(c["function"]["name"], "read");
        assert_eq!(c["function"]["arguments"], r#"{"path":"a.txt"}"#);
        assert!(tool_call("{broken", "x".into(), 0).is_none());
    }

    #[test]
    fn tool_result_takes_name_from_its_call() {
        let m = parse_messages(&json!([
            {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{\"b\":1,\"a\":2}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"}
        ]))
        .unwrap();
        assert_eq!(m[1].name.as_deref(), Some("read"));
        // テンプレートの sort_keys と同じ順に並ぶ
        assert_eq!(m[0].tool_calls[0].1.to_string(), r#"{"a":2,"b":1}"#);
    }
}
