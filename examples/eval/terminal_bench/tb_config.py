from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from omegaconf import OmegaConf

from examples.eval.eval_delegate import EvalEnvConfig


@dataclass
class TbEvalDatasetConfig:
    """Dataset configuration for Terminal-Bench runs."""

    name: str = ""
    dataset: str | None = None
    dataset_path: str | None = None
    dataset_config: str | None = None
    task_ids: list[str] = field(default_factory=list)
    exclude_task_ids: list[str] = field(default_factory=list)
    n_tasks: int | None = None
    agent: str | None = None
    model: str | None = None
    agent_kwargs: dict[str, Any] = field(default_factory=dict)
    n_concurrent: int | None = None
    n_attempts: int | None = None
    no_rebuild: bool | None = None
    cleanup: bool | None = None
    global_timeout_multiplier: float | None = None
    global_agent_timeout_sec: float | None = None
    global_test_timeout_sec: float | None = None

    def __post_init__(self):
        self.name = (self.name or "").strip()
        if not self.name:
            raise ValueError("Each TB dataset entry must include a non-empty `name`.")
        if not (self.dataset or self.dataset_path or self.dataset_config):
            raise ValueError(
                f"TB dataset '{self.name}' must define `dataset`, `dataset_path`, or `dataset_config`."
            )

    @classmethod
    def parse(cls, args, dataset_cfg: Mapping[str, Any], defaults: Mapping[str, Any]):
        cfg = OmegaConf.merge(
            OmegaConf.structured(cls),
            OmegaConf.create(defaults or {}),
            OmegaConf.create(dataset_cfg or {}),
        )
        obj = OmegaConf.to_object(cfg)
        if not isinstance(obj, cls):
            obj = cls(**obj)
        return obj

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for field_info in fields(self):
            value = getattr(self, field_info.name)
            if value in (None, {}, []):
                continue
            payload[field_info.name] = value
        return payload


@dataclass
class TbEvalEnvConfig(EvalEnvConfig):
    datasets: list[TbEvalDatasetConfig] = field(default_factory=list)

    @classmethod
    def parse(cls, args, raw_env_config: Mapping[str, Any], defaults: Mapping[str, Any]) -> "TbEvalEnvConfig":
        base_cfg: TbEvalEnvConfig = super().parse(raw_env_config, defaults)
        datasets = raw_env_config.get("datasets") or []
        base_cfg.datasets = [TbEvalDatasetConfig.parse(args, d, base_cfg.defaults) for d in datasets]
        return base_cfg


def build_tb_eval_env_config(args, raw_env_config: Mapping[str, Any], defaults: Mapping[str, Any]):
    return TbEvalEnvConfig.parse(args, raw_env_config, defaults)
