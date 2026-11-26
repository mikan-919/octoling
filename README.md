# octoling

日本語 | [English](README.en.md)

Thinking Machines Lab の言語モデル [Inkling-2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) の文章の部分を、NVIDIA の GPU 1枚で動かす推論エンジンです。Rust と CUDA で書いています。OpenAI 互換の HTTP API、ターミナルでの1回の質問、速さの測定の3つの使い方があります。

重みは2つの形式を読めます。1つは公式の safetensors（NVFP4 と MXFP8）、もう1つは [Strata](https://github.com/Niko1221/Strata) が配布する量子化済みの SQ-10（`.strata`、約 10.5 GB）です。16 GB の GPU では SQ-10 を使うと全体が VRAM に載ります。生成には、公式の下書きモデル（`mtp.safetensors`）による投機的デコード（小さなモデルが数トークン先を予想し、本体がそれを1回の計算でまとめて確かめる方法。採用するのは本体が確かめたトークンだけ）を使います。

開発と測定に使ったのは、RTX 5060 Ti 16 GB 1枚、Ryzen 7 7700、Windows 11 上の WSL2（割り当てた主記憶は約 25 GB）です。ほかの環境では確かめていません。

````console
$ octoling chat models/sq10 "Rustでフィボナッチ数列の10項目を求める関数を書いて" low
GPU に 12.38 GB 載せた (expert の枠 116 個)。残り 2.25 GB
下書きモデルに 0.50 GB 載せた。残り 1.72 GB
The user asks in Japanese: write a function in Rust to compute 10 items of Fibonacci sequence. ...（思考。端末では灰色）
Rustでフィボナッチ数列の最初の10項目（0,1,1,2,3,...）を返す関数例：

```rust
fn fibonacci_ten() -> Vec<u64> {
...
prompt 34 tokens (0 reused, prefill 738 ms), generated 348 tokens (111.9 tok/s, 2.40 tokens per verify, draft 3.0 ms + verify 18.3 ms per step), stop

[stop: プロンプト 34 トークン、生成 348 トークン、4.4 秒、78.3 tok/s]
````

`prompt` で始まる行の `111.9 tok/s` は生成だけの速さで、最後の行の `78.3 tok/s` は prefill（入力のトークン列をまとめて処理し、KV キャッシュ（注意の計算で使う過去のトークンのキーと値）を作る段階）を含めた速さです。

## 目次

- [性能](#性能)
- [必要な環境](#必要な環境)
- [組み立て](#組み立て)
- [重みの用意](#重みの用意)
- [使い方](#使い方)
- [環境変数](#環境変数)
- [仕組み](#仕組み)
- [制約](#制約)
- [問題への対処](#問題への対処)
- [開発](#開発)
- [参考にしたもの](#参考にしたもの)
- [ライセンス](#ライセンス)

## 性能

### 生成の速さ（Strata との比較）

条件：RTX 5060 Ti 16 GB・WSL2・SQ-10・貪欲法（温度 0）・各入力の後に 256 トークンを生成。Strata は 0.1.66 で、同じ SQ-10 と同じ下書きモデル（`--draft dspark`）を使いました。octoling は commit 2354069 の `spec_bench`（`OCTO_CTX=16384`）で測りました。GPU の温度の影響を減らすため、Strata と octoling を 60 秒の間を空けて交互に2回ずつ測っています。単位は tokens/s です。

| 入力 | 長さ（トークン） | Strata 1回目 | octoling 1回目 | Strata 2回目 | octoling 2回目 |
|---|---|---|---|---|---|
| コードの依頼 | 38 | 98.0 | 114.0 | 100.8 | 116.3 |
| 英語の文章の依頼 | 34 | 78.0 | 115.4 | 80.3 | 107.9 |
| 日本語の依頼 | 40 | 80.9 | 111.2 | 83.0 | 99.8 |
| 思考ありの依頼 | 56 | 94.8 | 138.4 | 98.2 | 142.1 |
| 連番のトークン ID（合成） | 1,000 | 78.9 | 303.2 | 86.9 | 311.5 |
| 文書の要約 | 3,203 | 104.3 | 116.5 | 111.6 | 120.3 |
| 連番のトークン ID（合成） | 8,000 | 106.3 | 140.2 | 114.2 | 132.7 |
| 文書 A | 3,091 | 95.6 | 134.8 | 92.8 | 128.8 |
| 文書 B | 2,885 | 96.8 | 132.0 | 91.4 | 133.5 |
| 文書 C | 2,498 | 93.9 | 115.7 | 91.6 | 119.9 |
| 文書 D | 3,463 | 90.2 | 132.6 | 94.9 | 132.7 |

測定に使った入力のファイルはこのリポジトリに入れていません。合成の入力は、続きを予想しやすいので下書きがよく当たります。

投機的デコードの速さは、生成される文によって変わります。計算の丸めが少し変わるだけで生成される文が変わり、下書きの当たり方も変わります。文書の要約（3,203 トークン）では、文が変わったときに 1 ステップで確定するトークン数が 2.67 から 2.03 に下がり、速さが約 95 tokens/s になった例がありました。同じ文を強制して比べると、この入力では octoling が 122〜131、Strata が 110 tokens/s でした。

prefill の expert の計算を int8 にした commit cd7fe40 では、prefill の丸めが変わったので、生成される文も変わりました。生成の計算は同じです。`spec_bench` で1回ずつ測ると、コードの依頼 119.4、日本語の依頼 107.7、文書の要約 99.7、文書 C 127.9 tokens/s でした（commit 2354069 では同じ測り方で 124.4、111.0、121.7、116.7）。文書の要約は、1 ステップで確定するトークン数が 2.67 から 1.90 に下がりました。

### prefill の速さ

同じ GPU で、`octoling bench`（commit cd7fe40）と Strata の `--bench --draft off` を交互に2回ずつ測りました（ミリ秒、短いほど速い）。Strata の時間は、出力の prefill の速さ（トークン/秒）から計算しました。

| 入力の長さ | Strata | octoling |
|---|---|---|
| 1,000 トークン | 514 / 500 | 407 / 407 |
| 3,203 トークン | 1,790 / 1,763 | 1,428 / 1,429 |
| 8,000 トークン | 4,641 / 4,644 | 3,526 / 3,525 |

### 公式の重み

公式の safetensors（19.5 GB）は 16 GB の VRAM に入らないので、一部の expert（混合専門家モデルで、トークンごとに選ばれる小さな MLP）を主記憶に置きます。下書きなしの生成は約 60 tokens/s でした。

## 必要な環境

| 項目 | 条件 |
|---|---|
| GPU | NVIDIA の計算能力 12.0 のカード（GeForce RTX 50 シリーズ）。カーネルは実行時に `sm_120` 向けに翻訳する |
| VRAM | SQ-10 で約 13 GB（文脈 8,192 トークン、下書きモデルを含む）。公式の重みは VRAM の空きに合わせて expert の置き場を決める |
| CPU | x86-64 で AVX-512 があるもの。無いと組み立てでエラーになる |
| 主記憶 | 重みのファイルを mmap する。公式の重みは 19.5 GB、SQ-10 は 10.5 GB。文脈を主記憶に残す分が既定で最大 4 GB |
| OS | Linux。確かめたのは WSL2 だけ |
| CUDA | ドライバと同じ版の NVRTC（実行時に CUDA C++ を翻訳するライブラリ）。flake では CUDA 13.3 の NVRTC を使う |
| Rust | 1.88 以降（edition 2024 と let チェーンを使う） |

## 組み立て

[Nix](https://nixos.org/) があれば、flake の開発用シェルに Rust と NVRTC が入ります。

```sh
git clone https://github.com/mikan-919/octoling
cd octoling
nix develop -c cargo build --release
```

実行ファイルは `target/release/octoling` にできます。flake の `LD_LIBRARY_PATH` は WSL2 の `libcuda.so` の場所（`/usr/lib/wsl/lib`）と NVRTC を指します。WSL2 ではない Linux では、`libcuda.so` のあるディレクトリをこの変数に足してください。

Nix を使わない場合は、Rust と、ドライバに合う版の NVRTC（`libnvrtc.so`）を用意し、`LD_LIBRARY_PATH` に `libcuda.so` と `libnvrtc.so` のディレクトリを入れてから組み立てます。ドライバより新しい NVRTC が作る PTX はドライバが読めません。

```sh
export LD_LIBRARY_PATH=/path/to/libcuda:/path/to/libnvrtc
cargo build --release
```

`.cargo/config.toml` で `target-cpu=native` を指定しているので、組み立てたマシンと同じ種類の CPU で使ってください。

## 重みの用意

重みのディレクトリには `tokenizer.json` と、次のどちらかを置きます。`mtp.safetensors` があれば下書きモデルとして読みます。以下の例は [Hugging Face の CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)（`hf`）を使います。

### SQ-10（16 GB の GPU 向け）

SQ-10 は Strata が配布する量子化済みの重みです。トークナイザーと下書きモデルは公式のリポジトリから取ります。

```sh
hf download strata-engine/Inkling-2-Mini-Strata Inkling-2-Mini-SQ-10.strata --local-dir models/sq10
hf download thinkingmachines/Inkling-2-Mini tokenizer.json mtp.safetensors --local-dir models/sq10
```

octoling が読めるのは `.strata` の v1.1 で、expert が S2_K・S3_K・SF4 の SQ-10 だけです。SQ-15 などほかの組み合わせは読めません。

### 公式の重み

```sh
hf download thinkingmachines/Inkling-2-Mini --local-dir models/official
```

ディレクトリに `model-00001-of-00001.safetensors` があれば公式の重みを、無ければ `.strata` のファイルを読みます。画像の部分の重みは使いません。

## 使い方

`octoling` を引数なしで実行すると、使い方が出ます。

```text
octoling serve <重みのディレクトリ> [待ち受けアドレス (既定 127.0.0.1:8080)]
octoling chat <重みのディレクトリ> <メッセージ> [reasoning_effort (既定 high)]
octoling bench <重みのディレクトリ> <トークンID,トークンID,...> [生成するトークン数]
```

### ターミナルで1回質問する（chat）

```sh
./target/release/octoling chat models/sq10 "Rustの所有権を短く説明して" low
```

答えを流して表示し、思考は灰色で出します。サンプリングは温度 1.0、種は時刻から決めます。3つ目の引数は思考の量（[reasoning_effort](#reasoning_effort)）です。

### OpenAI 互換のサーバー（serve）

```sh
./target/release/octoling serve models/sq10 127.0.0.1:8080
```

```text
GPU に 12.38 GB 載せた (expert の枠 116 個)。残り 2.25 GB
下書きモデルに 0.50 GB 載せた。残り 1.71 GB
http://127.0.0.1:8080/v1 で待ち受け中 (POST /v1/chat/completions, GET /v1/models)
```

エンドポイントは `GET /v1/models` と `POST /v1/chat/completions` の2つです。モデルの ID は `inkling-2-mini` です。

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"日本の首都は？一言で"}],"reasoning_effort":"none","temperature":0,"max_tokens":512}'
```

```json
{"choices":[{"finish_reason":"stop","index":0,"message":{"content":"東京","reasoning_content":"The user asks in Japanese: \"日本の首都は？一言で\" ... Final: 東京.","role":"assistant"}}],"created":1795533966,"id":"chatcmpl-17955339666e8750","model":"inkling-2-mini","object":"chat.completion","usage":{"completion_tokens":101,"prompt_tokens":22,"total_tokens":123}}
```

思考は `reasoning_content`、答えは `content` に入ります。`reasoning_effort` が `none` でもモデルは短く考えるので、`max_tokens` が小さいと答えの前に止まります（`finish_reason` が `length`）。

サーバーは各リクエストのあとに、標準エラーへ次のような1行を出します。`reused` は前のリクエストの文脈を使い回したトークン数です。

```text
prompt 22 tokens (10 reused, prefill 58 ms), generated 101 tokens (126.2 tok/s, 2.30 tokens per verify, draft 3.0 ms + verify 14.9 ms per step), stop
```

#### 受け付ける項目

| 項目 | 説明 | 既定 |
|---|---|---|
| `messages` | `system`・`user`・`assistant`・`tool` のメッセージ。`content` は文字列か、文章の部分の配列。`assistant` の `reasoning_content` と `tool_calls` も読む | 必須 |
| `reasoning_effort` | 思考の量。下の表 | `high`（0.9） |
| `max_tokens`・`max_completion_tokens` | 生成するトークン数の上限 | 文脈長の上限 |
| `temperature`・`top_p`・`top_k` | サンプリングの設定。`temperature` が 0 なら貪欲法 | 1.0・1.0・0（使わない） |
| `seed` | 乱数の種 | 時刻から決める |
| `stream` | `true` で Server-Sent Events で流す | `false` |
| `stream_options.include_usage` | 流す時に、最後にトークン数を送る | `false` |
| `tools` | 関数の定義。モデルの呼び出しは `tool_calls` で返し、`finish_reason` は `tool_calls` | なし |
| `tool_choice` | `none` だけを見る（ツールを渡さない）。ほかの値は `auto` と同じ扱い | `auto` |

<a name="reasoning_effort"></a>`reasoning_effort` は文字列か 0〜0.99 の数で、プロンプトの `Thinking effort level` に入ります。

| 文字列 | 値 |
|---|---|
| `none` | 0.0 |
| `minimal` | 0.1 |
| `low` | 0.2 |
| `medium` | 0.7 |
| `high` | 0.9 |
| `xhigh`・`max` | 0.99 |

#### 文脈の使い回し

GPU は文脈を1つだけ持ちます。次のリクエストのプロンプトが GPU にある文脈の続きなら、増えた部分だけを prefill します。そうでない時は、主記憶に残した過去のプロンプトの文脈（合計で最大 `OCTO_KV_GB`、既定 4 GB）から、最も長く一致するものを GPU に戻してから続けます。チャットの次の番のように、前と同じ始まりのプロンプトで効きます。

リクエストは1つずつ順に処理します。同時に来たリクエストは、前のものが終わるまで待ちます。後ろのリクエストが止まらないように、30 秒間読み書きが進まない接続は切ります。

### 速さを測る（bench）

`bench` は prefill と、下書きなしの貪欲法の生成を測ります。入力はトークン ID の列で、`@ファイル名` でファイルから読めます。`;` で区切ると、モデルを1回読み込んだまま複数の入力を続けて測ります。`serve` と同じ形のプロンプトのトークン ID は、例の `prompt_ids` で作れます。

```sh
nix develop -c cargo run --release --example prompt_ids -- models/sq10/tokenizer.json high "Rustの所有権を説明して" > prompt.txt
./target/release/octoling bench models/sq10 @prompt.txt 128
```

```text
GPU に 12.38 GB 載せた (expert の枠 116 個)。残り 2.44 GB
prefill 19 tokens: 66 ms, 286.0 tok/s
tok 200010 -> 200001:27.942 9850:8.904 2986:7.256 20698:7.141 18057:7.123 (66 ms, first [-2.0983 -4.4991 -1.8066] sum -2325645.5000)
generated 128 tokens: 98.3 tok/s
200001,200008,371,150409,171960,...
```

2行目は prefill の時間、3行目は最後の入力トークンの次の候補の上位5つ（トークン ID とロジット）、最後の行は生成したトークン ID です。投機的デコードの速さは、例の `spec_bench` で測ります（[開発](#開発)）。

## 環境変数

| 変数 | 意味 | 既定 |
|---|---|---|
| `OCTO_CTX` | 文脈長の上限（トークン）。公式の重みでは 8,192 を超えると、global 層（文脈の全体を見る注意の層）の KV を主記憶に置き、注意を CPU で計算する（1 トークンに 32 KB）。この時は下書きモデルを使わない。SQ-10 では KV を VRAM に置き、16 GB の GPU で 65,536 まで入ることを確かめた | 8192 |
| `OCTO_DRAFT` | `0` で下書きモデルを使わない。公式の重みでは `1` の時だけ使う（確かめる計算が遅く、下書きの分を取り返せないことが多いため） | SQ-10 は使う |
| `OCTO_TAU` | 下書きを止めるしきい値。下書きの第1候補の確率の積がこれを下回ったら、そこで下書きを止める。実際のしきい値は、それまでの当たり方で 0.5〜2 倍に補正する | 0.35 |
| `OCTO_KV_GB` | serve が主記憶に残す文脈の合計の上限（GB） | 4 |
| `OCTO_EXPERT_SLOTS` | 公式の重みで、主記憶の expert を VRAM に置く枠の数（最大 255） | VRAM の空きから決める |
| `OCTO_THREADS` | CPU の計算のスレッド数 | 論理コア数の半分 |
| `OCTO_REPEAT` | bench・spec_bench で入力の列を繰り返す回数 | 1 |
| `OCTO_STEP` | bench で prefill を使わず、1トークンずつ処理する | なし |
| `OCTO_CPU` | bench で GPU を使わず CPU だけで処理する | なし |

開発用の変数は [開発](#開発) にあります。

## 仕組み

### 処理の流れ

Inkling-2 Mini の文章の部分は 24 層で、そのうち 6〜17 層を3回繰り返して通ります（合わせて 48 回の層の通過）。各層の MLP は 128 個の expert から、トークンごとに 8 個を選びます。

- **prefill:** 最大 1,536 トークンをまとめて、テンソルコア（GPU の行列積専用の演算器）で計算します。重みを読む回数をトークン数で割れるので、計算の速さが上限になります。
- **生成:** 1トークンごとの計算は、重みをメモリから読む速さが上限になります。カーネルの列を CUDA Graph（GPU の処理の並びを記録して、まとめて再生する仕組み）に記録して再生し、起動の待ち時間を減らしています。
- **投機的デコード:** 下書きモデル（DSpark。本体の途中の層の出力を入力にする3層のモデル）が最大 7 トークン先を予想します。本体は最大 8 トークンを1回で計算し、予想と一致した所までを確定します。8 トークン以下の計算は、重みを1回読むあいだに全トークン分を掛ける専用のカーネルで行います。

### 重みの形式と置き場

| | 公式の safetensors | SQ-10 |
|---|---|---|
| 大きさ | 19.5 GB | 10.5 GB |
| expert | NVFP4（0〜19 層）と MXFP8（20〜23 層） | S2_K・S3_K・SF4（2〜4 ビット） |
| 本体の行列 | BF16 | 多くは int8（行ごとのスケール）、global 層の QKV は BF16 |
| 出力層 | BF16 | 6 ビット（S6_K） |
| 置き場 | MXFP8 の expert は主記憶に置き、VRAM の枠に載っている分は GPU で、残りは CPU で計算する | 全部 VRAM |

SQ-10 の expert は、量子化したまま GPU で展開してテンソルコアに渡します。prefill では、S2_K と S3_K の重みを int8 に直し、活性も int8 にして（gate/up の入力は 64 要素ごと、down の入力は 32 要素ごとのスケール）、int8 のテンソルコアで掛けます。Strata の expert のカーネルも、64 要素ごとに int8 にした活性を受け取ります。CPU 側の計算（公式の重みの一部の expert、長い文脈の注意）は AVX-512 で行います。

### ソースの構成

| ファイル | 内容 |
|---|---|
| `src/main.rs` | コマンドの入口 |
| `src/serve.rs` | チャットのテンプレート、サンプリング、ツール呼び出しの解釈、HTTP サーバー |
| `src/gpu.rs`・`src/gpu.cu` | GPU の順伝播と CUDA のカーネル |
| `src/gpu/draft.rs` | 下書きモデル DSpark |
| `src/gpu/host_attn.rs` | 主記憶の KV に対する注意（CPU） |
| `src/model.rs` | CPU だけの順伝播（GPU 版と比べる基準） |
| `src/strata.rs` | `.strata` の読み込み |
| `src/st.rs`・`src/nvfp4.rs`・`src/kernels.rs` | safetensors の読み込み、NVFP4 の展開、AVX-512 の内積 |

## 制約

- 文章だけを扱います。`content` に画像などの文章以外の部分があると HTTP 400 を返します。
- モデルは Inkling-2 Mini だけです。
- リクエストは1つずつ処理します。
- サンプリングは `temperature`・`top_p`・`top_k`・`seed` だけです。`stop`・`n`・`logprobs`・`response_format` などは無視します。
- `tool_choice` の `required` と関数名の指定は、`auto` と同じに扱います。
- 投機的デコードの出力は、下書きなしの貪欲法の出力と同じとは限りません。確かめる計算（複数トークン）と1トークンずつの計算で足し算の順番が違い、第1候補と第2候補の差が小さい所で選ぶトークンが変わることがあります。
- 認証はありません。外から接続できるアドレスで待ち受ける時は注意してください。

## 問題への対処

| 症状 | 確かめること |
|---|---|
| 組み立てで `AVX-512 が必要` | CPU に AVX-512 があるか。`.cargo/config.toml` の `target-cpu=native` を消していないか |
| 起動時に `Unable to dynamically load the "nvrtc" shared library` | `LD_LIBRARY_PATH` に `libnvrtc.so` のディレクトリがあるか。Nix では `nix develop -c` を付けて実行する |
| カーネルの読み込みで失敗する | NVRTC の版がドライバより新しくないか。GPU が計算能力 12.0 か |
| `に safetensors も .strata もない` | 重みのディレクトリに `model-00001-of-00001.safetensors` か `.strata` があるか |
| `tokenizer.json を読めない` | 重みのディレクトリに `tokenizer.json` があるか |
| 起動時に VRAM が足りない | ほかのプロセスが VRAM を使っていないか。`OCTO_CTX` を小さくするか、`OCTO_DRAFT=0` で下書きモデルを読まない |
| HTTP 400 `文脈長の上限を超えている` | プロンプトが `OCTO_CTX` 以上の長さ。`OCTO_CTX` を大きくする |
| 答えが空で `finish_reason` が `length` | 思考で `max_tokens` を使い切っている。`max_tokens` を大きくするか、`reasoning_effort` を下げる |

## 開発

```sh
nix develop -c cargo test --release
```

単体のテストは CPU だけで動き、GPU のカーネルが翻訳できるかも確かめます。モデルを使う確認は例（`examples/`）で行います。CI（GitHub Actions）では、テストに加えて `cargo fmt --check` と `cargo clippy --release --all-targets -- -D warnings` も実行します。

| 例 | 内容 |
|---|---|
| `spec_test <重み> <トークン ID のファイル>` | 投機的デコードの部品（確かめる計算と巻き戻し、KV）が1トークンずつの計算と一致するかを確かめる。`MATCH` と `kv MATCH` が出れば一致 |
| `spec_bench <重み> <入力;入力...> <トークン数>` | 下書きなしの貪欲法と投機的デコードの速さ、1ステップの内訳（下書き・確かめ・後処理の時間）、確定したトークン数の分布を出す。`OCTO_FORCE=ファイル` で生成する文を固定し、同じ文で速さを比べられる |
| `prompt_ids <tokenizer.json> <reasoning_effort> <本文>` | serve と同じ形のプロンプトのトークン ID を出す |

```sh
nix develop -c cargo build --release --examples
nix develop -c target/release/examples/spec_bench models/sq10 @prompt.txt 256
```

開発用の環境変数：

| 変数 | 意味 |
|---|---|
| `OCTO_STAMP` | 生成の CUDA Graph のカーネルごとに GPU の時刻を記録し、カーネルの種類別の時間を出す |
| `OCTO_PROF` | 層や区切りごとに GPU と CPU の時間を出す |
| `OCTO_TRACE` | CPU 版の各層の出力を、llama.cpp の `l_out-N` と比べられる形で出す |
| `OCTO_FORCE` | spec_bench で、本体が選んだトークンをファイルのトークン ID の列に置き換える |

## 参考にしたもの

- 順伝播の仕様は [llama.cpp](https://github.com/ggml-org/llama.cpp) の `src/models/inkling2.cpp` と `inkling2-loop.cpp` に合わせました。
- `.strata` の形式は [Strata](https://github.com/Niko1221/Strata)（MIT）の `strata_reader.cpp`・`ink_manifest.h`・`sk_codecs.cpp` に合わせました。
- 下書きモデル DSpark の構造は [Xolotl](https://github.com/Indras-Mirror/Xolotl)（MIT）の `src/program/inkling_dspark.cpp` と、llama.cpp の `src/models/dspark.cpp` に合わせました。

## ライセンス

[MIT License](LICENSE) です。モデルの重みには、それぞれの配布元のライセンスと利用規定が適用されます。
