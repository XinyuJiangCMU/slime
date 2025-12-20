# Terminal Bench Eval (Slime)

This folder wires Terminal Bench (TB) into Slime as an eval delegate. The TB
run happens on the host via the `tb` CLI, and Slime reads back `accuracy` and
`n_resolved`.

This guide is written for ML/algorithm folks who just want it to run.

## What runs where

- Slime runs your training/eval loop.
- Slime calls the TB delegate client.
- The TB delegate server (`tb_server.py`) runs `tb run ...` on the host.
- The server reads the latest TB JSON results and returns metrics to Slime.

## Prereqs

1) A working OpenAI-compatible inference endpoint, e.g.:
   - `http://127.0.0.1:30002/v1`

2) Terminal Bench installed and its `tb` CLI available.
   - Activate your TB venv first:
     ```bash
     source terminal-bench/.venv/bin/activate
     tb --help
     ```

3) A Slime eval config file that includes `eval.datasets`.
   - Slime requires at least one dataset under `eval.datasets`.
   - You can reuse your existing eval config; just add the delegate section.

## Step 1: Start the TB server

Run on the host (same machine where `tb` works):

```bash
python slime/examples/eval/terminal_bench/tb_server.py \
  --host 0.0.0.0 --port 9050 \
  --output-root /tmp/tb-eval
```

What it does:
- Uses `OPENAI_API_KEY=EMPTY`
- Runs `tb run -a terminus-2 -m openai/<model> ... --n-concurrent 8`
- Waits for completion, then returns `accuracy` and `n_resolved`

## Step 2: Configure Slime eval

You need an eval config. Example:

```yaml
eval:
  # Slime still needs normal eval datasets (can be any small one).
  datasets:
    - name: aime
      path: /root/datasets/aime-2024/aime-2024.jsonl
      rm_type: math

  # TB delegate config.
  delegate:
    - name: terminal_bench
      url: http://localhost:9050         # "/evaluate" auto-added if missing
      timeout_secs: 1200                 # 20 minutes
      model_name: qwen3-8b
      api_base: http://127.0.0.1:30002/v1
      n_tasks: 10
      n_concurrent: 8
```

Notes:
- `model_name` is auto-normalized to `openai/<model>` if you omit the prefix.
- `n_concurrent` is currently fixed to 8 in `tb_server.py`.
- The TB client auto-adds `/evaluate` if you give a bare host:port.

## Step 3: Tell Slime to use the delegate rollout

Add this to your training/eval command:

```bash
--eval-config /path/to/your_eval_config.yaml \
--eval-function-path examples.eval.eval_delegate_rollout.generate_rollout
```

This makes Slime call the TB delegate during evaluation.

## Quick sanity check (eval-only)

If you just want to verify the TB integration, run a quick eval-only pass
(you still need your normal Slime args for model/data/etc.):

```bash
python slime/train.py \
  --num-rollout 0 \
  --eval-interval 1 \
  --eval-config /path/to/your_eval_config.yaml \
  --eval-function-path examples.eval.eval_delegate_rollout.generate_rollout \
  ...other required args...
```

## Common gotchas

- 404 from TB server: use `url: http://localhost:9050` or `.../evaluate`.
- Timeouts: keep `timeout_secs` large (TB tasks can compile code).
- No TB metrics: check `/tmp/tb-eval/<run_id>/` for JSON results.

## Reference: the CLI command it runs

The server is aligned with:

```bash
OPENAI_API_KEY=EMPTY tb run -a terminus-2 -m openai/qwen3-8b \
  --agent-kwarg api_base=http://127.0.0.1:30002/v1 \
  --n-concurrent 8
```
