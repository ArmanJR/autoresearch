# (auto) autoresearch

This is the search, on top of Karpathy's hyperparameter search. Basically, it’s a search for a search.

## My Idea

Karpathy's autoresearch is brilliant. It works, but there is a catch: the "Blank Page Problem" of AI. LLMs are fundamentally reactive engines. They are designed to follow, not to lead. If the leader (the user) is asleep, the follower (the agent) defaults to the "average" of all its knowledge, which is mediocre and boring, and gravitates toward safe, incremental moves.

But move 37 was neither safe nor average.

To fix this, we need another piece of software that acts as a "Chaos Monkey" or a "Creative Director." Think of it as the mutation rate in a genetic algorithm; without it, you can get stuck in a local minimum forever.

This repo is my attempt to build that monkey. This is a search for a search.

## Architecture

Each experiment method lives in its own directory with `train.py`, `prepare.py`, and `program.md`. The `baseline/` directory is Karpathy's vanilla autoresearch loop. Other directories (e.g. `mad-scientist/`) add a **director** — a Go binary that generates creative research directives before each experiment iteration. The director works in three steps:

1. **Summarize** the current `train.py` via DeepSeek Chat (so it always knows the actual code state)
2. **Fetch** a random ML paper abstract from arxiv (external novelty injection)
3. **Generate** a directive via DeepSeek Reasoner, combining code summary + experiment history + paper into one specific idea

The output is a tentative suggestion ("I think you could try...") so the upstream agent reasons about it rather than blindly executing.

> I chose Go because I wanted the running agent to have absolutely zero context about the director and to see it as a black box. Just a binary that spits out ideas.

### Structure

```
baseline/                 # vanilla autoresearch (control group)
  train.py, prepare.py, program.md

mad-scientist/            # experiment with director-driven exploration
  train.py, prepare.py, program.md, director, .env

director/                 # director source code
  main.go
  configs/*.json          # per-experiment config (prompts, arxiv terms, model, temp)
  logs/api_calls.jsonl    # centralized API call log across all experiments
  .env                    # DEEPSEEK_API_KEY

results/                  # tracking (gitignored)
  <method>/<run-id>/results.tsv

analysis.ipynb            # cross-method comparison (envelope, terminal perf, stall)
Makefile
```

### Current Experiments

| Name | Director | Description |
|------|----------|-------------|
| `baseline` | None | Vanilla autoresearch. The agent decides what to try next on its own. Control group. |
| `mad-scientist` | DeepSeek Reasoner (temp 1.2) | Summarizes current code, reads experiment history, fetches a random ML paper from arxiv, then generates a bold directive framed as a suggestion. Combines code awareness + historical context + external novelty. |

### Commands

```bash
# List available director configs
make list

# Build + deploy director for an experiment (default: linux/arm64 for NVIDIA Jetson)
make deploy EXPERIMENT=mad-scientist

# Build for local macOS instead
make deploy EXPERIMENT=mad-scientist GOOS=darwin GOARCH=arm64

# Run the director (from inside an experiment directory)
./director --verbose

# Run training (from inside an experiment directory)
python3 train.py
```

### Adding a new experiment method

1. Copy `baseline/` to a new directory
2. Create `director/configs/<name>.json` with custom system prompt, user prompt, arxiv terms, model, temperature
3. `make deploy EXPERIMENT=<name>`
4. Edit `program.md` in the new directory to integrate the director into the loop

## Hardware

I'm GPU poor. I only have an NVIDIA Jetson AGX Orin to play with. Let's port autoresearch to that.

### Jetson AGX Orin port

This fork has been adapted to run on an **NVIDIA Jetson AGX Orin 32GB** (JetPack 6, CUDA 12.6, PyTorch 2.10). The key changes from upstream:

- **Replaced Flash Attention 3** with PyTorch's built-in `scaled_dot_product_attention` (FA3/kernels package targets Hopper/desktop Ampere and doesn't build on Jetson's SM 8.7).
- **Removed `torch.compile`** — Triton is not available on aarch64, so the inductor backend fails. Model and optimizer run in eager mode.
- **Removed `kernels` dependency** and the pinned torch CUDA index from `pyproject.toml` (Jetson uses its own JetPack-provided PyTorch).
- **Tuned hyperparameters** for the Orin's ~5–24 TFLOPS BF16 throughput (size-dependent) and 30 GB unified memory.

#### Setup on Jetson

```bash
# Install deps (torch is already provided by JetPack)
pip3 install rustbpe tiktoken pyarrow requests numpy pandas matplotlib

# Clone and prep
git clone <repo-url> && cd autoresearch
python3 prepare.py    # download data + train tokenizer
python3 train.py      # baseline run (~5 min)
```

#### Hyperparameter sweep results

All runs use the fixed 5-minute time budget on a Jetson AGX Orin 32GB with `MAX_SEQ_LEN=512`, `HEAD_DIM=64`, `WINDOW_PATTERN="L"`:

| DEPTH | DEVICE_BATCH_SIZE | TOTAL_BATCH_SIZE | val_bpb | Steps | Params | VRAM |
|-------|-------------------|------------------|---------|-------|--------|------|
| 4 | 16 | 2^15 | 1.488 | 872 | 11.5M | 1.2 GB |
| 8 | 32 | 2^17 | 1.519 | 95 | 50.3M | 6.2 GB |
| 6 | 32 | 2^17 | 1.419 | 147 | 26.3M | 4.4 GB |
| 6 | 32 | 2^16 | 1.357 | 279 | 26.3M | 4.4 GB |
| 6 | 32 | 2^15 | 1.341 | 535 | 26.3M | 4.4 GB |
| **6** | **32** | **2^14** | **1.338** | **1018** | **26.3M** | **4.3 GB** |

The best configuration is **DEPTH=6, DEVICE_BATCH_SIZE=32, TOTAL_BATCH_SIZE=2^14** (the current defaults in this fork). Key takeaways:

- **DEPTH=8 is too large** — the 50M param model only gets 95 steps, not enough to converge in 5 minutes.
- **DEPTH=6 (26.3M params) is the sweet spot** — enough capacity while still allowing hundreds of steps.
- **Smaller batch sizes win** — on this hardware, more optimizer steps matters more than bigger batches. The improvement plateaus around 2^14 (grad_accum=1).
- **Only 4.3 GB VRAM used** out of 30 GB available, leaving plenty of room for the autonomous agent to experiment with larger architectures.

---

# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069).

## How it works

The repo is deliberately kept small and only really has a three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

## Quick start

**Requirements:** A single NVIDIA GPU (tested on H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.

## Platform support

This code currently requires that you have a single NVIDIA GPU. In principle it is quite possible to support CPU, MPS and other platforms but this would also bloat the code. I'm not 100% sure that I want to take this on personally right now. People can reference (or have their agents reference) the full/parent nanochat repository that has wider platform support and shows the various solutions (e.g. a Flash Attention 3 kernels fallback implementation, generic device support, autodetection, etc.), feel free to create forks or discussions for other platforms and I'm happy to link to them here in the README in some new notable forks section or etc.

Seeing as there seems to be a lot of interest in tinkering with autoresearch on much smaller compute platforms than an H100, a few extra words. If you're going to try running autoresearch on smaller computers (Macbooks etc.), I'd recommend one of the forks below. On top of this, here are some recommendations for how to tune the defaults for much smaller models for aspiring forks:

1. To get half-decent results I'd use a dataset with a lot less entropy, e.g. this [TinyStories dataset](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean). These are GPT-4 generated short stories. Because the data is a lot narrower in scope, you will see reasonable results with a lot smaller models (if you try to sample from them after training).
2. You might experiment with decreasing `vocab_size`, e.g. from 8192 down to 4096, 2048, 1024, or even - simply byte-level tokenizer with 256 possibly bytes after utf-8 encoding.
3. In `prepare.py`, you'll want to lower `MAX_SEQ_LEN` a lot, depending on the computer even down to 256 etc. As you lower `MAX_SEQ_LEN`, you may want to experiment with increasing `DEVICE_BATCH_SIZE` in `train.py` slightly to compensate. The number of tokens per fwd/bwd pass is the product of these two.
4. Also in `prepare.py`, you'll want to decrease `EVAL_TOKENS` so that your validation loss is evaluated on a lot less data.
5. In `train.py`, the primary single knob that controls model complexity is the `DEPTH` (default 8, here). A lot of variables are just functions of this, so e.g. lower it down to e.g. 4.
6. You'll want to most likely use `WINDOW_PATTERN` of just "L", because "SSSL" uses alternating banded attention pattern that may be very inefficient for you. Try it.
7. You'll want to lower `TOTAL_BATCH_SIZE` a lot, but keep it powers of 2, e.g. down to `2**14` (~16K) or so even, hard to tell.

I think these would be the reasonable hyperparameters to play with. Ask your favorite coding agent for help and copy paste them this guide, as well as the full source code.

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)

## License

MIT
