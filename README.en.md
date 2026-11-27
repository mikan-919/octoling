# octoling

[日本語](README.md) | English

[![CI](https://github.com/mikan-919/octoling/actions/workflows/ci.yml/badge.svg)](https://github.com/mikan-919/octoling/actions/workflows/ci.yml)

An inference engine that runs the text part of Thinking Machines Lab's [Inkling-2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on a single NVIDIA GPU. Written in Rust and CUDA. It has three modes: an OpenAI-compatible HTTP API, a one-shot question in the terminal, and a benchmark.

It reads two weight formats: the official safetensors (NVFP4 and MXFP8), and SQ-10 (`.strata`, about 10.5 GB), a quantized format distributed by [Strata](https://github.com/Niko1221/Strata). With SQ-10 the whole model fits in VRAM on a 16 GB GPU. Generation uses speculative decoding: a small draft model guesses a few tokens ahead, the main model checks them in one pass, and only the tokens the main model confirms are kept. The draft model is either [DFlash2](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) from Inco AI (`dflash2.safetensors`) or the official DSpark (`mtp.safetensors`).

Development and measurements were done on one RTX 5060 Ti 16 GB, a Ryzen 7 7700, and WSL2 on Windows 11 (about 25 GB of RAM assigned). Other environments have not been tested.

This English README covers performance, requirements, building, and usage. Environment variables, internals, troubleshooting, and development notes are in the [Japanese README](README.md).

````console
$ octoling chat models/sq10 "Rustでフィボナッチ数列の10項目を求める関数を書いて" low
GPU に 11.47 GB 載せた (expert の枠 118 個、global 層の KV は f16)。残り 4.15 GB
下書きモデル (DFlash2) に 0.65 GB 載せた。残り 2.95 GB
The user asks in Japanese: write a function in Rust to compute 10 items of Fibonacci sequence. ...（reasoning, shown in gray in the terminal）
Rustでフィボナッチ数列の最初の10項目（0,1,1,2,3,...）を返す関数例：

```rust
fn fibonacci_ten() -> Vec<u64> {
...
prompt 34 tokens (0 reused, prefill 738 ms), generated 348 tokens (111.9 tok/s, 2.40 tokens per verify, draft 3.0 ms + verify 18.3 ms per step), stop

[stop: プロンプト 34 トークン、生成 348 トークン、4.4 秒、78.3 tok/s]
````

The prompt asks, in Japanese, for a Rust function that computes the first 10 Fibonacci numbers. Log messages are in Japanese. `111.9 tok/s` on the `prompt` line is the generation speed alone; `78.3 tok/s` on the last line includes prefill (processing the input tokens in one batch to build the KV cache).

## Contents

- [Performance](#performance)
- [Requirements](#requirements)
- [Building](#building)
- [Getting the weights](#getting-the-weights)
- [Usage](#usage)
- [Limitations](#limitations)
- [License](#license)

## Performance

### Draft models (DSpark and DFlash2)

Setup: RTX 5060 Ti 16 GB, WSL2, SQ-10, greedy decoding (temperature 0), 256 tokens generated after each input. Measured with `spec_bench` at commit c966b8a; DSpark was run with `OCTO_DRAFT=dspark`. The four inputs were made with `prompt_ids` (reasoning_effort in parentheses): a code request "Rustでフィボナッチ数列の10項目を求める関数を書いて" (low), an English writing request "Write a short essay about why the sky is blue, for a curious ten-year-old." (low), a Japanese request "日本の四季について、それぞれの特徴を説明してください" (low), and a reasoning request (an English word problem about when one train catches up with another, high).

| Input | Tokens confirmed per step, DSpark | DFlash2 | tokens/s, DSpark | DFlash2 |
|---|---|---|---|---|
| Code request | 2.67 | 3.82 | 132.0 | 178.2 |
| English writing request | 1.72 | 2.31 | 98.1 | 119.3 |
| Japanese request | 1.42 | 2.02 | 87.7 | 113.8 |
| Request with reasoning | 4.13 | 4.65 | 178.4 | 205.8 |

Greedy decoding without a draft model ran at 88–95 tokens/s. Drafting takes about 3.2 ms per step with either model. VRAM use is 0.50 GB for DSpark and 0.65 GB for DFlash2. Comparing the draft cutoff `OCTO_TAU` at 0.15, 0.25, 0.35, and 0.5, the default 0.35 was the fastest or close to it on all four inputs with DFlash2 as well.

### Generation speed (compared with Strata)

The octoling numbers in this section were measured with DSpark.

Setup: RTX 5060 Ti 16 GB, WSL2, SQ-10, greedy decoding (temperature 0), 256 tokens generated after each input. Strata is version 0.1.66 with the same SQ-10 weights and the same draft model (`--draft dspark`). octoling was measured with `spec_bench` at commit 2354069 (`OCTO_CTX=16384`). To reduce the effect of GPU temperature, Strata and octoling were run alternately twice each, 60 seconds apart. Unit: tokens/s.

| Input | Length (tokens) | Strata run 1 | octoling run 1 | Strata run 2 | octoling run 2 |
|---|---|---|---|---|---|
| Code request | 38 | 98.0 | 114.0 | 100.8 | 116.3 |
| English writing request | 34 | 78.0 | 115.4 | 80.3 | 107.9 |
| Japanese request | 40 | 80.9 | 111.2 | 83.0 | 99.8 |
| Request with reasoning | 56 | 94.8 | 138.4 | 98.2 | 142.1 |
| Sequential token IDs (synthetic) | 1,000 | 78.9 | 303.2 | 86.9 | 311.5 |
| Document summary | 3,203 | 104.3 | 116.5 | 111.6 | 120.3 |
| Sequential token IDs (synthetic) | 8,000 | 106.3 | 140.2 | 114.2 | 132.7 |
| Document A | 3,091 | 95.6 | 134.8 | 92.8 | 128.8 |
| Document B | 2,885 | 96.8 | 132.0 | 91.4 | 133.5 |
| Document C | 2,498 | 93.9 | 115.7 | 91.6 | 119.9 |
| Document D | 3,463 | 90.2 | 132.6 | 94.9 | 132.7 |

The input files are not in this repository. Synthetic inputs are easy to predict, so the draft model hits often.

Speculative decoding speed depends on the generated text. A small change in rounding can change the text, which changes how often the draft hits. On the document summary (3,203 tokens), one change in the text dropped the tokens confirmed per step from 2.67 to 2.03 and the speed to about 95 tokens/s. With the same text forced, octoling ran at 122–131 and Strata at 110 tokens/s on this input.

Commit cd7fe40 moved the prefill expert computation to int8, which changed prefill rounding and therefore the generated text. The generation computation itself is unchanged. Single `spec_bench` runs gave 119.4 (code request), 107.7 (Japanese request), 99.7 (document summary), and 127.9 (document C) tokens/s, against 124.4, 111.0, 121.7, and 116.7 at commit 2354069 with the same method. On the document summary, tokens confirmed per step dropped from 2.67 to 1.90.

### Prefill speed

On the same GPU, `octoling bench` (commit cd7fe40) and Strata's `--bench --draft off` were run alternately twice each (milliseconds, lower is faster). Strata's time is computed from the prefill speed (tokens/s) it reports.

| Input length | Strata | octoling |
|---|---|---|
| 1,000 tokens | 514 / 500 | 407 / 407 |
| 3,203 tokens | 1,790 / 1,763 | 1,428 / 1,429 |
| 8,000 tokens | 4,641 / 4,644 | 3,526 / 3,525 |

### Long context (FP8 KV)

When the context limit (`OCTO_CTX`) is above 8,192, the KV cache of the global layers is kept in FP8 (E4M3). `OCTO_KV=f16` or `OCTO_KV=fp8` overrides this; with the official weights, KV kept in RAM is always f16. The KV per token goes from 32 KB to 16 KB, and a 131,072-token context fits on a 16 GB GPU together with the draft model.

Setup: RTX 5060 Ti 16 GB, WSL2, SQ-10, DFlash2, greedy decoding, 256 tokens generated after each input. The input is English text joined from the WikiText-103 validation set. Unit: tokens/s.

| Input length | KV | Greedy | Speculative |
|---|---|---|---|
| 30,000 tokens | f16 | 75.9 | 87.2 |
| 30,000 tokens | FP8 | 81.8 | 97.9 |
| 130,000 tokens | FP8 | 56.7 | 60.2 |

At 130,000 tokens, VRAM use was 13.35 GB for the main model and 0.65 GB for DFlash2, with 0.59 GB left. Prefilling 30,000 tokens took 14.1 s with f16 and 14.5 s with FP8.

Output was compared with f16 at the 256 positions after 30,000 tokens. The mean KL divergence (difference between two probability distributions; 0 means identical) was 0.0021, the top-1 token matched at 98.0% of positions, and the negative log-likelihood of the correct token was 1.633 for f16 and 1.627 for FP8. The KL divergence between SQ-10 and the official weights is about 0.05 (Strata's published figure). NVFP4 (an E4M3 scale per 16 elements) gave a KL divergence of 0.0051 and a top-1 match of 95.3%.

```sh
OCTO_KV=f16 nix develop -c target/release/examples/kv_quality models/sq10 @ids.txt 30000 256 f16.bin
OCTO_KV=fp8 nix develop -c target/release/examples/kv_quality models/sq10 @ids.txt 30000 256 fp8.bin
python examples/kv_compare.py f16.bin fp8.bin @ids.txt 30000
```

### Official weights

The official safetensors (19.5 GB) do not fit in 16 GB of VRAM, so some experts (the small MLPs in the mixture-of-experts model, selected per token) stay in RAM. Generation without the draft model ran at about 60 tokens/s.

## Requirements

| Item | Requirement |
|---|---|
| GPU | NVIDIA, compute capability 12.0 (GeForce RTX 50 series). Kernels are compiled for `sm_120` at runtime |
| VRAM | About 12 GB with SQ-10 (8,192-token context, including the draft model), about 14 GB with a 131,072-token context. With the official weights, expert placement follows the free VRAM |
| CPU | x86-64 with AVX-512. The build fails without it |
| RAM | Weight files are mmapped: 19.5 GB for the official weights, 10.5 GB for SQ-10. Up to 4 GB more by default for contexts kept in RAM by `serve` |
| OS | Linux. Only WSL2 has been tested |
| CUDA | NVRTC (the library that compiles CUDA C++ at runtime) of the same version as the driver. The flake uses NVRTC from CUDA 13.3 |
| Rust | 1.88 or later (edition 2024 and let chains) |

## Building

With [Nix](https://nixos.org/), the flake's dev shell provides Rust and NVRTC.

```sh
git clone https://github.com/mikan-919/octoling
cd octoling
nix develop -c cargo build --release
```

The binary is `target/release/octoling`. The flake's `LD_LIBRARY_PATH` points to the locations of `libcuda.so` on WSL2 (`/usr/lib/wsl/lib`) and NixOS (`/run/opengl-driver/lib`), and to NVRTC. On other Linux systems, add the directory that contains `libcuda.so` to this variable.

Without Nix, install Rust and an NVRTC (`libnvrtc.so`) that matches your driver, put the directories of `libcuda.so` and `libnvrtc.so` in `LD_LIBRARY_PATH`, and build. The driver cannot load PTX produced by an NVRTC newer than itself.

```sh
export LD_LIBRARY_PATH=/path/to/libcuda:/path/to/libnvrtc
cargo build --release
```

`.cargo/config.toml` sets `target-cpu=native`, so run the binary on the same kind of CPU it was built on.

## Getting the weights

The weight directory holds `tokenizer.json` and one of the two formats below. For the draft model, `dflash2.safetensors` (DFlash2) is loaded if present, otherwise `mtp.safetensors` (DSpark). Set `OCTO_DRAFT=dspark` to use DSpark even when `dflash2.safetensors` is present, or `OCTO_DRAFT=0` to use no draft model. The examples use the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) (`hf`).

### SQ-10 (for 16 GB GPUs)

SQ-10 is a quantized format distributed by Strata. The tokenizer and DSpark come from the official repository, and DFlash2 from Inco AI's repository. DFlash2's file is named `model.safetensors`, so rename it to `dflash2.safetensors`.

```sh
hf download strata-engine/Inkling-2-Mini-Strata Inkling-2-Mini-SQ-10.strata --local-dir models/sq10
hf download thinkingmachines/Inkling-2-Mini tokenizer.json mtp.safetensors --local-dir models/sq10
hf download incoai/Inkling-2-Mini-DFlash2 model.safetensors --local-dir models/dflash2
mv models/dflash2/model.safetensors models/sq10/dflash2.safetensors
```

To skip DFlash2, leave out the last two lines.

octoling reads only `.strata` v1.1 with SQ-10 experts (S2_K, S3_K, SF4). Other combinations such as SQ-15 are not supported.

(Experimental) If the directory also holds `experts.tq`, the experts it contains are computed from trellis-quantized weights (each weight is computed from a short window of a bit string, with no value table) instead of SQ-10's S2_K/S3_K, and the `.strata` may leave those experts out. An `experts.tq` at about 2.1 bits per expert weight (6.41 GB) plus a `.strata` without those experts and without the unused DSpark weights (1.66 GB) comes to 8.07 GB, with a lower KL divergence from the official weights than SQ-10 on English, code and Japanese text. Generation runs at about 90% of SQ-10's speed. The tools that build `experts.tq` are not in the repository yet.

### Official weights

```sh
hf download thinkingmachines/Inkling-2-Mini --local-dir models/official
```

If the directory has `model-00001-of-00001.safetensors`, the official weights are loaded; otherwise the `.strata` file is. Vision weights are not used.

## Usage

Running `octoling` with no arguments prints the usage.

```text
octoling serve <weight dir> [listen address (default 127.0.0.1:8080)]
octoling chat <weight dir> <message> [reasoning_effort (default high)]
octoling bench <weight dir> <token ID,token ID,...> [tokens to generate]
```

### One question in the terminal (chat)

```sh
./target/release/octoling chat models/sq10 "Explain Rust ownership briefly" low
```

The answer is streamed, with reasoning in gray. Sampling uses temperature 1.0 with a seed from the clock. The third argument is the amount of reasoning ([reasoning_effort](#reasoning_effort)).

### OpenAI-compatible server (serve)

```sh
./target/release/octoling serve models/sq10 127.0.0.1:8080
```

The endpoints are `GET /v1/models` and `POST /v1/chat/completions`. The model ID is `inkling-2-mini`.

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Capital of Japan? One word"}],"reasoning_effort":"none","temperature":0,"max_tokens":512}'
```

Reasoning is returned in `reasoning_content` and the answer in `content`. Even with `reasoning_effort` set to `none` the model reasons briefly, so a small `max_tokens` can stop generation before the answer (`finish_reason` is `length`).

#### Request fields

| Field | Description | Default |
|---|---|---|
| `messages` | `system`, `user`, `assistant`, and `tool` messages. `content` is a string or an array of text parts. `reasoning_content` and `tool_calls` on `assistant` messages are read | required |
| `reasoning_effort` | Amount of reasoning. See below | `high` (0.9) |
| `max_tokens`, `max_completion_tokens` | Maximum tokens to generate | context limit |
| `temperature`, `top_p`, `top_k` | Sampling. `temperature` 0 means greedy | 1.0, 1.0, 0 (off) |
| `seed` | Random seed | from the clock |
| `stream` | `true` streams with Server-Sent Events | `false` |
| `stream_options.include_usage` | When streaming, send token counts at the end | `false` |
| `tools` | Function definitions. Calls from the model are returned in `tool_calls` with `finish_reason` `tool_calls` | none |
| `tool_choice` | Only `none` is honored (tools are not passed). Other values act as `auto` | `auto` |

<a name="reasoning_effort"></a>`reasoning_effort` is a string or a number from 0 to 0.99, inserted into the prompt as `Thinking effort level`.

| String | Value |
|---|---|
| `none` | 0.0 |
| `minimal` | 0.1 |
| `low` | 0.2 |
| `medium` | 0.7 |
| `high` | 0.9 |
| `xhigh`, `max` | 0.99 |

#### Context reuse

The GPU holds one context. If the next prompt continues the context on the GPU, only the new part is prefilled. Otherwise the longest matching context among past prompts kept in RAM (up to `OCTO_KV_GB` in total, 4 GB by default) is copied back to the GPU first. This helps when prompts share the same beginning, such as the next turn of a chat.

Requests are processed one at a time. Concurrent requests wait until the previous one finishes. So that later requests are not blocked, a connection that makes no read or write progress for 30 seconds is closed.

### Benchmark (bench)

`bench` measures prefill and greedy generation without the draft model. The input is a list of token IDs; `@file` reads it from a file, and `;` separates several inputs measured with the model loaded once. The `prompt_ids` example produces token IDs in the same prompt format as `serve`.

```sh
nix develop -c cargo run --release --example prompt_ids -- models/sq10/tokenizer.json high "Explain Rust ownership" > prompt.txt
./target/release/octoling bench models/sq10 @prompt.txt 128
```

Speculative decoding speed is measured with the `spec_bench` example (see the development section of the [Japanese README](README.md#開発)).

## Limitations

- Text only. A `content` part that is not text returns HTTP 400.
- Inkling-2 Mini only.
- One request at a time.
- Only `temperature`, `top_p`, `top_k`, and `seed` affect sampling. `stop`, `n`, `logprobs`, `response_format`, and others are ignored.
- `tool_choice` set to `required` or to a function name acts as `auto`.
- Speculative decoding output is not always identical to greedy output without the draft model. The multi-token verify pass and the single-token pass add in a different order, so the chosen token can differ where the top two candidates are close.
- There is no authentication. Be careful when listening on an address reachable from outside.

## License

[MIT License](LICENSE). The model weights are subject to the licenses and usage policies of their distributors.
