#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shlex
import subprocess
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from flask import Flask, jsonify, request
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from examples.eval.terminal_bench.tb_config import TbEvalDatasetConfig

logger = logging.getLogger("tb_eval_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@dataclass
class EvalRequestPayload:
    rollout_id: int
    router_url: str
    defaults: dict[str, Any] = field(default_factory=dict)
    benchmarks: list[TbEvalDatasetConfig] = field(default_factory=list)


def _openai_api_base(router_url: str) -> str:
    return router_url.rstrip("/") + "/v1"


def _pass_at_k_estimator(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    prod = 1.0
    for value in range(n - c + 1, n + 1):
        prod *= 1.0 - k / value
    return float(1.0 - prod)


def _compute_pass_at_k(results: list[dict[str, Any]]) -> dict[int, float]:
    task_counts: dict[str, list[int]] = defaultdict(list)
    for result in results:
        task_id = result.get("task_id")
        if not task_id:
            continue
        task_counts[task_id].append(1 if result.get("is_resolved") else 0)

    if not task_counts:
        return {}

    min_attempts = min(len(counts) for counts in task_counts.values())
    if min_attempts < 2:
        return {}

    k_values = {2**i for i in range(1, int(math.log2(min_attempts)) + 1)}
    if min_attempts >= 5:
        k_values.add(5)
    if min_attempts >= 10:
        k_values.add(10)

    pass_at_k: dict[int, float] = {}
    for k in sorted(k_values):
        passes = []
        for success in task_counts.values():
            if len(success) < k:
                continue
            passes.append(_pass_at_k_estimator(len(success), sum(success), k))
        if passes:
            pass_at_k[k] = float(sum(passes) / len(passes))
    return pass_at_k


def _summarize_results(results_json: dict[str, Any]) -> dict[str, float]:
    results = results_json.get("results") or []
    if not isinstance(results, list):
        return {}

    total = len(results)
    resolved = sum(1 for r in results if r.get("is_resolved"))
    unresolved = total - resolved
    accuracy = (resolved / total) if total else 0.0

    metrics: dict[str, float] = {
        "n_total": float(total),
        "n_resolved": float(resolved),
        "n_unresolved": float(unresolved),
        "accuracy": float(accuracy),
    }

    pass_at_k = _compute_pass_at_k(results)
    for k, v in pass_at_k.items():
        metrics[f"pass_at_{k}"] = float(v)

    if total:
        total_in = sum((r.get("total_input_tokens") or 0) for r in results)
        total_out = sum((r.get("total_output_tokens") or 0) for r in results)
        metrics["avg_input_tokens"] = float(total_in) / total
        metrics["avg_output_tokens"] = float(total_out) / total

    return metrics


@dataclass
class ServerConfig:
    output_root: Path
    tb_command: str = "tb"
    tb_workdir: Path | None = None
    default_agent: str = "terminus-1"
    default_model: str = "openai/slime-openai-model"
    default_n_concurrent: int = 4
    default_n_attempts: int = 1

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ServerConfig":
        return cls(
            output_root=Path(args.output_root).expanduser().resolve(),
            tb_command=args.tb_command,
            tb_workdir=Path(args.tb_workdir).expanduser().resolve() if args.tb_workdir else None,
            default_agent=args.default_agent,
            default_model=args.default_model,
            default_n_concurrent=args.default_n_concurrent,
            default_n_attempts=args.default_n_attempts,
        )


class TerminalBenchEvaluator:
    def __init__(self, config: ServerConfig):
        self._config = config
        self._lock = threading.Lock()
        self._config.output_root.mkdir(parents=True, exist_ok=True)

    def evaluate(self, payload: EvalRequestPayload) -> dict[str, Any]:
        if not payload.benchmarks:
            warning_msg = "No TB benchmarks specified in delegate config; skipping evaluation."
            logger.warning(warning_msg)
            return {
                "job_id": uuid.uuid4().hex,
                "command": None,
                "output_dir": None,
                "log_path": None,
                "warning": warning_msg,
                "raw_metrics": {},
            }

        job_id = uuid.uuid4().hex
        raw_metrics: dict[str, Any] = {"tb": {}}
        runs: list[dict[str, Any]] = []

        with self._lock:
            for benchmark in payload.benchmarks:
                result = self._run_single_benchmark(payload, benchmark)
                runs.append(result["run_info"])
                raw_metrics["tb"][benchmark.name] = result["metrics"]

        command_summary = "\n".join(run["command"] for run in runs) if runs else None
        log_path = runs[-1]["log_path"] if runs else None
        output_dir = runs[-1]["output_dir"] if runs else None

        return {
            "job_id": job_id,
            "command": command_summary,
            "output_dir": output_dir,
            "log_path": log_path,
            "raw_metrics": raw_metrics,
            "runs": runs,
        }

    def _run_single_benchmark(self, payload: EvalRequestPayload, benchmark: TbEvalDatasetConfig) -> dict[str, Any]:
        run_id = f"rollout-{payload.rollout_id}-{benchmark.name}-{int(time.time())}"
        run_dir = self._config.output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "tb_eval.log"

        command = self._build_command(payload.defaults, benchmark, run_id, payload.router_url)
        env = self._build_env()
        logger.info("Starting TB eval for %s: %s", benchmark.name, " ".join(shlex.quote(p) for p in command))
        self._run_command(command, env=env, log_path=log_path)

        metrics = self._collect_metrics(run_dir)
        return {
            "run_info": {
                "benchmark": benchmark.name,
                "command": " ".join(shlex.quote(part) for part in command),
                "output_dir": str(run_dir),
                "log_path": str(log_path),
                "run_id": run_id,
            },
            "metrics": metrics,
        }

    def _build_command(
        self,
        defaults: Mapping[str, Any],
        benchmark: TbEvalDatasetConfig,
        run_id: str,
        router_url: str,
    ) -> list[str]:
        cmd = shlex.split(self._config.tb_command)
        cmd += [
            "run",
            "--output-path",
            str(self._config.output_root),
            "--run-id",
            run_id,
        ]

        if benchmark.dataset:
            cmd += ["--dataset", benchmark.dataset]
        if benchmark.dataset_path:
            cmd += ["--dataset-path", benchmark.dataset_path]
        if benchmark.dataset_config:
            cmd += ["--dataset-config", benchmark.dataset_config]

        for task_id in benchmark.task_ids:
            cmd += ["--task-id", task_id]
        for task_id in benchmark.exclude_task_ids:
            cmd += ["--exclude-task-id", task_id]
        if benchmark.n_tasks is not None:
            cmd += ["--n-tasks", str(benchmark.n_tasks)]

        agent = benchmark.agent or defaults.get("agent") or self._config.default_agent
        model = benchmark.model or defaults.get("model") or self._config.default_model
        cmd += ["--agent", agent]
        if model:
            cmd += ["--model", model]

        n_concurrent = benchmark.n_concurrent or defaults.get("n_concurrent") or self._config.default_n_concurrent
        n_attempts = benchmark.n_attempts or defaults.get("n_attempts") or self._config.default_n_attempts
        cmd += ["--n-concurrent", str(n_concurrent)]
        cmd += ["--n-attempts", str(n_attempts)]

        no_rebuild = benchmark.no_rebuild if benchmark.no_rebuild is not None else defaults.get("no_rebuild")
        cleanup = benchmark.cleanup if benchmark.cleanup is not None else defaults.get("cleanup")

        if no_rebuild is True:
            cmd.append("--no-rebuild")
        if no_rebuild is False:
            cmd.append("--rebuild")
        if cleanup is True:
            cmd.append("--cleanup")
        if cleanup is False:
            cmd.append("--no-cleanup")

        if benchmark.global_timeout_multiplier is not None:
            cmd += ["--global-timeout-multiplier", str(benchmark.global_timeout_multiplier)]
        if benchmark.global_agent_timeout_sec is not None:
            cmd += ["--global-agent-timeout-sec", str(benchmark.global_agent_timeout_sec)]
        if benchmark.global_test_timeout_sec is not None:
            cmd += ["--global-test-timeout-sec", str(benchmark.global_test_timeout_sec)]

        agent_kwargs = self._normalize_agent_kwargs(defaults.get("agent_kwargs"))
        agent_kwargs += self._normalize_agent_kwargs(benchmark.agent_kwargs)

        api_base = _openai_api_base(router_url)
        if not any(arg.startswith("api_base=") for arg in agent_kwargs):
            agent_kwargs.append(f"api_base={api_base}")

        for kwarg in agent_kwargs:
            cmd += ["--agent-kwarg", kwarg]

        return cmd

    @staticmethod
    def _normalize_agent_kwargs(value: Any) -> list[str]:
        if isinstance(value, dict):
            return [f"{k}={v}" for k, v in value.items()]
        if isinstance(value, list):
            return [str(v) for v in value]
        return []

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.setdefault("OPENAI_API_KEY", "EMPTY")
        return env

    def _run_command(self, cmd: list[str], *, env: dict[str, str], log_path: Path):
        with open(log_path, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(self._config.tb_workdir) if self._config.tb_workdir else None,
            )
            retcode = process.wait()
        if retcode != 0:
            with open(log_path, encoding="utf-8", errors="ignore") as log_file:
                tail = "".join(log_file.readlines()[-200:])
            raise RuntimeError(f"`tb run` failed with exit code {retcode}. See {log_path}\n{tail}")

    @staticmethod
    def _collect_metrics(run_dir: Path) -> dict[str, float]:
        results_path = run_dir / "results.json"
        if not results_path.exists():
            logger.warning("Results file missing at %s", results_path)
            return {}
        try:
            results_json = json.loads(results_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse %s: %s", results_path, exc)
            return {}
        return _summarize_results(results_json)


def build_app(evaluator: TerminalBenchEvaluator) -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health_check():
        return jsonify({"status": "ok"})

    @app.post("/evaluate")
    def evaluate_endpoint():
        try:
            raw_payload = request.get_json(force=True, silent=False)
            cfg = OmegaConf.merge(
                OmegaConf.structured(EvalRequestPayload),
                OmegaConf.create(raw_payload or {}),
            )
            payload = OmegaConf.to_object(cfg)
            result = evaluator.evaluate(payload)
            return jsonify(result)
        except OmegaConfBaseException as exc:
            logger.exception("Invalid request payload")
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:  # noqa: BLE001
            logger.exception("Evaluation failed")
            return jsonify({"error": str(exc)}), 500

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Terminal-Bench evaluation HTTP server.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9060)
    parser.add_argument(
        "--output-root",
        type=str,
        default="./tb-eval-output",
        help="Directory to store TB run outputs.",
    )
    parser.add_argument(
        "--tb-command",
        type=str,
        default="tb",
        help="Command used to invoke Terminal-Bench (e.g. 'tb' or 'python -m terminal_bench.cli.tb.main').",
    )
    parser.add_argument(
        "--tb-workdir",
        type=str,
        default=None,
        help="Working directory to run TB from (useful if you want relative paths).",
    )
    parser.add_argument("--default-agent", type=str, default="terminus-1")
    parser.add_argument("--default-model", type=str, default="openai/slime-openai-model")
    parser.add_argument("--default-n-concurrent", type=int, default=4)
    parser.add_argument("--default-n-attempts", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    config = ServerConfig.from_args(args)
    evaluator = TerminalBenchEvaluator(config)
    app = build_app(evaluator)
    logger.info(
        "Starting TB evaluation server on %s:%s (output root=%s)",
        args.host,
        args.port,
        config.output_root,
    )
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()