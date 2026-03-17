# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
2. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `python3 prepare.py`.
3. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
4. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `python3 train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The results file lives at a path provided by the `RESULTS_TSV` environment variable. If not set, default to `results.tsv` in the current directory.

The TSV has a header row and 7 columns:

```
iter	val_bpb	best_val_bpb	memory_gb	status	description	timestamp
```

1. iteration number, starting at 1 (sequential counter, incremented for every experiment including crashes)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. best_val_bpb so far — the running minimum of val_bpb across all kept experiments up to this point (carry forward the previous best on discard/crash)
4. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried
7. timestamp in ISO 8601 format (e.g. 2026-03-09T22:15:03)

Example:

```
iter	val_bpb	best_val_bpb	memory_gb	status	description	timestamp
1	0.997900	0.997900	44.0	keep	baseline	2026-03-09T22:15:03
2	0.993200	0.993200	44.2	keep	increase LR to 0.04	2026-03-09T22:20:45
3	1.005000	0.993200	44.0	discard	switch to GeLU activation	2026-03-09T22:26:12
4	0.000000	0.993200	0.0	crash	double model width (OOM)	2026-03-09T22:31:58
```

## The experiment loop

### First run (baseline)

Before experimenting, establish the baseline:

1. Run `python3 train.py > run.log 2>&1` without any modifications.
2. Read the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
3. Record iter=1 in results.tsv with status `keep` and description `baseline`.
4. Create the checkpoint: `cp train.py train.py.best`

### Experiment loop

After the baseline, LOOP FOREVER:

1. **Ask your supervisor for direction**: Run `./director` (this may take up to a few minutes, be patient) — it's how your supervisor professor talks to you. He observes your work, and will give you a suggestion for what to try next. Read his comment carefully, then decide: do you agree with the direction? Does it make sense given the current state of the code? You can follow it closely, adapt the core idea, combine it with your own intuition, or ignore it entirely if it's bad advice. It's your call — you're the one writing the code and your judgment matters. The supervisor gives direction; you decide how (or whether) to apply it.
2. Modify `train.py` with your chosen experimental idea.
3. Run the experiment: `python3 train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
4. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
5. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
6. Record the results in the tsv.
7. If val_bpb improved (lower): save the current code as the new best with `cp train.py train.py.best`
8. If val_bpb is equal or worse, or it crashed: restore the last good version with `cp train.py.best train.py`

The idea is that you are a researcher working under a supervisor. The supervisor (`./director`) proposes research directions, but you are the hands-on expert. You understand the code, you see the results, and you make the final call on every change. If the supervisor's idea is brilliant, run with it. If it's half-baked, extract what's useful and adapt it. If it's nonsense, ignore it and try something better. The supervisor will have a fresh idea for you every iteration.

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, restore from `train.py.best`, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. Run `./director` every iteration for fresh ideas — it will always have a new suggestion for you. The loop runs until the human interrupts you, period.

**NEVER CHANGE YOUR WORKING DIRECTORY**: Stay in the current directory.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
