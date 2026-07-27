#!/usr/bin/env python
"""Fail-closed audit of a completed CounterStrike-1K Dust2 MIRA pilot run.

The training and evaluation launchers intentionally write ordinary JSON, JSONL, YAML, TSV, and
SHA-256 provenance files. This command checks those artifacts against the preregistered protocol
and emits one compact, machine-readable audit report. It does not recompute model metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

EXPECTED_FULL_MANIFEST_SHA256 = "e6d1199595327ccbed7a79760266f1c30fdcf23b717232705c9b11bb6d8707d3"
EXPECTED_SELECTION_SHA256 = "5b4733ba910c96221b06923699b6595b27b23ec9659cb1b15032bbf7dceb9cf9"
EXPECTED_SPLITS = {
    "train": {
        "matches": 39,
        "rounds": 835,
        "pov_rows": 8350,
        "aligned_pov_hours": 95.38576388888887,
    },
    "val": {
        "matches": 3,
        "rounds": 54,
        "pov_rows": 540,
        "aligned_pov_hours": 4.959461805555556,
    },
    "test": {
        "matches": 3,
        "rounds": 52,
        "pov_rows": 520,
        "aligned_pov_hours": 5.3564236111111105,
    },
}
EXPECTED_PIPELINE_STATES = [
    ("codec", "running"),
    ("codec", "complete"),
    ("single", "running"),
    ("single", "complete"),
    ("shared", "running"),
    ("shared", "complete"),
    ("pipeline", "complete"),
]
EXPECTED_PRIMARY_SEEDS = [37, 38, 39]
EXPECTED_ACTION_SEEDS = [37, 38, 39, 40, 41]
EXPECTED_ACTION_MODES = ["true", "batch-shifted", "time-shifted", "zero"]
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"{path} does not contain a JSON object")
    return payload


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"{path} does not contain a YAML mapping")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_results(payload: dict[str, Any], source: Path) -> None:
    results = payload.get("results")
    if not isinstance(results, dict) or not results:
        raise ValueError(f"{source}: missing non-empty results mapping")
    nonfinite = [
        key
        for key, value in results.items()
        if not isinstance(value, (int, float)) or not math.isfinite(float(value))
    ]
    if nonfinite:
        raise ValueError(f"{source}: non-finite result metrics: {nonfinite}")


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@dataclass
class Auditor:
    root: Path
    expected_training_commit: str | None = None
    expected_evaluator_commit: str | None = None
    checks: dict[str, Any] = field(default_factory=dict)

    def require(self, name: str, condition: bool, evidence: Any) -> None:
        if not condition:
            raise ValueError(f"{name}: {evidence}")
        self.checks[name] = evidence

    def audit_dataset(self) -> None:
        source = self.root / "provenance" / "dataset.json"
        payload = _read_json(source)
        self.require("dataset.map", payload.get("map_slug") == "dust2", payload.get("map_slug"))
        self.require(
            "dataset.full_manifest_sha256",
            payload.get("full_manifest_sha256") == EXPECTED_FULL_MANIFEST_SHA256,
            payload.get("full_manifest_sha256"),
        )
        self.require(
            "dataset.selection_sha256",
            payload.get("selection_sha256") == EXPECTED_SELECTION_SHA256,
            payload.get("selection_sha256"),
        )
        self.require(
            "dataset.source_shards", payload.get("source_shards") == 116, payload.get("source_shards")
        )
        self.require(
            "dataset.materialization",
            payload.get("verification") == {"samples": 9410, "payload_files": 47050},
            payload.get("verification"),
        )
        statistics = payload["statistics"]
        self.require(
            "dataset.match_disjoint",
            all(not overlap for overlap in statistics["match_overlap"].values()),
            statistics["match_overlap"],
        )
        for split, expected in EXPECTED_SPLITS.items():
            observed = statistics["splits"][split]
            for key, value in expected.items():
                if isinstance(value, float):
                    valid = math.isclose(float(observed[key]), value, rel_tol=0.0, abs_tol=1e-10)
                else:
                    valid = observed[key] == value
                self.require(f"dataset.{split}.{key}", valid, observed[key])

    def audit_code_provenance(self) -> str:
        provenance = self.root / "provenance"
        commit = (provenance / "code_commit.txt").read_text().strip()
        self.require("training_code.commit_format", bool(COMMIT_RE.fullmatch(commit)), commit)
        if self.expected_training_commit is not None:
            self.require(
                "training_code.expected_commit",
                commit == self.expected_training_commit,
                {"expected": self.expected_training_commit, "observed": commit},
            )
        self.require(
            "training_code.clean_status",
            (provenance / "code_status.txt").read_text() == "",
            (provenance / "code_status.txt").read_text(),
        )
        self.require(
            "training_code.clean_patch",
            (provenance / "code.patch").stat().st_size == 0,
            (provenance / "code.patch").stat().st_size,
        )
        for filename in (
            "environment_files.sha256",
            "installed_packages.txt",
            "nvidia_smi_q.txt",
            "uname.txt",
        ):
            path = provenance / filename
            self.require(f"provenance.{filename}", path.stat().st_size > 0, path.stat().st_size)
        return commit

    def audit_pipeline_status(self) -> dict[str, datetime]:
        rows: list[tuple[str, str, str]] = []
        for line in (self.root / "pipeline_status.tsv").read_text().splitlines():
            timestamp, stage, state = line.split("\t")
            rows.append((timestamp, stage, state))
        observed = [(stage, state) for _, stage, state in rows]
        self.require("pipeline.states", observed == EXPECTED_PIPELINE_STATES, observed)
        times = {f"{stage}.{state}": _timestamp(timestamp) for timestamp, stage, state in rows}
        self.require(
            "pipeline.monotonic_timestamps",
            all(
                _timestamp(left[0]) <= _timestamp(right[0])
                for left, right in zip(rows, rows[1:], strict=False)
            ),
            [row[0] for row in rows],
        )
        return times

    @staticmethod
    def _inner_config(config: dict[str, Any], arm: str) -> dict[str, Any]:
        model = config["model"]["architecture"]["config"]
        return model if arm == "single" else model["wm_config"]

    @staticmethod
    def _dataloader_config(config: dict[str, Any]) -> dict[str, Any]:
        dataloader = config["dataloader"]
        return {
            "num_workers": dataloader["num_workers"],
            "shuffle_buffer_size": dataloader["shuffle_buffer_size"],
            "prefetch_factor": dataloader.get("prefetch_factor", 2),
            "persistent_workers": dataloader.get("persistent_workers", False),
            "pin_memory": dataloader.get("pin_memory"),
        }

    def audit_configs(self) -> dict[str, int]:
        codec = _read_yaml(self.root / "codec" / "codec_config.yaml")
        single = _read_yaml(self.root / "single" / "world_model_config.yaml")
        shared = _read_yaml(self.root / "shared" / "world_model_config.yaml")

        self.require("codec.deterministic", codec["run"]["deterministic"] is True, codec["run"])
        self.require("codec.duration_hours", codec["run"]["max_duration_hours"] == 1.0, codec["run"])
        self.require("codec.map", codec["dataset"]["map_slug"] == "dust2", codec["dataset"])

        expected_arm = {
            "single": {"config": single, "batch_size": 10, "n_players": 1, "group_mode": "single"},
            "shared": {
                "config": shared,
                "batch_size": 1,
                "n_players": 10,
                "group_mode": "synchronized",
            },
        }
        for arm, expected in expected_arm.items():
            config = expected["config"]
            run = config["run"]
            dataset = config["dataset"]
            self.require(f"{arm}.seed", run["seed"] == 28, run["seed"])
            self.require(f"{arm}.deterministic", run["deterministic"] is True, run["deterministic"])
            self.require(f"{arm}.duration_hours", run["max_duration_hours"] == 5.5, run["max_duration_hours"])
            self.require(f"{arm}.batch_size", run["batch_size"] == expected["batch_size"], run["batch_size"])
            self.require(f"{arm}.map", dataset["map_slug"] == "dust2", dataset["map_slug"])
            self.require(
                f"{arm}.n_players", dataset["n_players"] == expected["n_players"], dataset["n_players"]
            )
            self.require(
                f"{arm}.group_mode",
                dataset["group_mode"] == expected["group_mode"],
                dataset["group_mode"],
            )
            self.require(f"{arm}.train_split", dataset["train_split"] == "train", dataset["train_split"])
            self.require(f"{arm}.test_split", dataset["test_split"] == "val", dataset["test_split"])

        single_inner = self._inner_config(single, "single")
        shared_inner = self._inner_config(shared, "shared")
        comparable_keys = (
            "actions",
            "video",
            "codec_checkpoint",
            "patch_size",
            "n_register_tokens",
            "attention_gating",
            "n_context_frames",
            "dropout_action_prob",
            "max_mouse_movement",
            "learned_temporal_pool",
            "causal",
            "use_codec_posterior_mean",
            "activation_checkpointing",
            "ada_attn_ln",
            "psd_loss_prob",
            "psd_weight",
            "use_clean_past",
            "hidden_dim",
            "n_head",
            "n_kv_head",
            "n_layers",
            "time_attention_every",
        )
        differences = {
            key: {"single": single_inner.get(key), "shared": shared_inner.get(key)}
            for key in comparable_keys
            if single_inner.get(key) != shared_inner.get(key)
        }
        self.require("arms.identical_inner_model", not differences, differences)
        self.require(
            "arms.identical_optimizer",
            single["optim"] == shared["optim"],
            {"single": single["optim"], "shared": shared["optim"]},
        )
        single_dataloader = self._dataloader_config(single)
        shared_dataloader = self._dataloader_config(shared)
        self.require(
            "arms.identical_dataloader",
            single_dataloader == shared_dataloader,
            {"single": single_dataloader, "shared": shared_dataloader},
        )

        single_frames = (
            single["run"]["batch_size"] * single["dataset"]["n_players"] * single_inner["video"]["timesteps"]
        )
        shared_frames = (
            shared["run"]["batch_size"] * shared["dataset"]["n_players"] * shared_inner["video"]["timesteps"]
        )
        single_actions = single["run"]["batch_size"] * single["dataset"]["n_players"]
        shared_actions = shared["run"]["batch_size"] * shared["dataset"]["n_players"]
        self.require(
            "arms.frames_per_step", single_frames == shared_frames == 160, [single_frames, shared_frames]
        )
        self.require(
            "arms.action_streams_per_step",
            single_actions == shared_actions == 10,
            [single_actions, shared_actions],
        )
        return {"single": single_frames, "shared": shared_frames}

    def audit_training_stage(self, stage: str, hours: float, frames_per_step: int) -> dict[str, Any]:
        stage_root = self.root / stage
        rows = [
            json.loads(line)
            for line in (stage_root / "metrics.jsonl").read_text().splitlines()
            if line.strip()
        ]
        train = [row for row in rows if row.get("kind") == "train"]
        validation = [row for row in rows if row.get("kind") == "validation"]
        limits = [row for row in rows if row.get("kind") == "time_limit"]
        self.require(f"{stage}.time_limit_count", len(limits) == 1, len(limits))
        limit = limits[0]
        elapsed = float(limit["elapsed_wall_seconds"])
        self.require(
            f"{stage}.timed_duration",
            hours * 3600 <= elapsed <= hours * 3600 + 30,
            elapsed,
        )
        self.require(f"{stage}.has_train_metrics", bool(train), len(train))
        self.require(f"{stage}.has_validation", bool(validation), len(validation))
        self.require(
            f"{stage}.initial_validation",
            any(row["step"] == 0 for row in validation),
            [row["step"] for row in validation],
        )
        self.require(
            f"{stage}.post_initial_validation",
            any(row["step"] > 0 for row in validation),
            [row["step"] for row in validation],
        )
        train_steps = [int(row["step"]) for row in train]
        self.require(
            f"{stage}.monotonic_train_steps",
            all(left < right for left, right in zip(train_steps, train_steps[1:], strict=False)),
            train_steps[-5:],
        )
        final_step = int(limit["step"])
        checkpoint_root = stage_root / f"checkpoint-{final_step}"
        checkpoint = checkpoint_root / "checkpoint.pth"
        training_state = checkpoint_root / "training_state.pth"
        self.require(f"{stage}.final_checkpoint", checkpoint.stat().st_size > 0, checkpoint.stat().st_size)
        self.require(
            f"{stage}.final_training_state",
            training_state.stat().st_size > 0,
            training_state.stat().st_size,
        )
        last_train = train[-1]
        expected_frames = (int(last_train["step"]) + 1) * frames_per_step
        self.require(
            f"{stage}.processed_frame_accounting",
            int(last_train["System/n_frames_processed"]) == expected_frames,
            {
                "expected": expected_frames,
                "observed": int(last_train["System/n_frames_processed"]),
            },
        )
        return {
            "time_limit_step": final_step,
            "elapsed_wall_seconds": elapsed,
            "last_logged_processed_frames": int(last_train["System/n_frames_processed"]),
            "checkpoint": str(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": _sha256(checkpoint),
        }

    def audit_gpu_telemetry(self, times: dict[str, datetime]) -> dict[str, Any]:
        source = self.root / "provenance" / "gpu_timeseries.csv"
        with source.open(newline="") as stream:
            rows = list(csv.DictReader(stream, skipinitialspace=True))
        self.require("gpu_telemetry.rows", bool(rows), len(rows))
        timestamp_key = "timestamp"
        memory_key = "memory.used [MiB]"
        parsed = [
            (
                datetime.strptime(row[timestamp_key], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc),
                int(row[memory_key].split()[0]),
            )
            for row in rows
        ]
        shared_start = times["shared.running"]
        pipeline_end = times["pipeline.complete"]
        shared_rows = [memory for timestamp, memory in parsed if shared_start <= timestamp <= pipeline_end]
        self.require(
            "gpu_telemetry.shared_coverage", bool(shared_rows), [str(parsed[0][0]), str(parsed[-1][0])]
        )
        return {
            "first_timestamp": parsed[0][0].isoformat(),
            "last_timestamp": parsed[-1][0].isoformat(),
            "peak_memory_mib_observed": max(memory for _, memory in parsed),
            "shared_peak_memory_mib": max(shared_rows),
            "single_trace_is_partial": parsed[0][0] > times["single.running"],
        }

    def _audit_eval_provenance(self, root: Path) -> str:
        provenance = root / "provenance"
        commit = (provenance / "evaluator_code_commit.txt").read_text().strip()
        self.require(f"{root.name}.evaluator_commit_format", bool(COMMIT_RE.fullmatch(commit)), commit)
        if self.expected_evaluator_commit is not None:
            self.require(
                f"{root.name}.expected_evaluator_commit",
                commit == self.expected_evaluator_commit,
                {"expected": self.expected_evaluator_commit, "observed": commit},
            )
        self.require(
            f"{root.name}.clean_evaluator_status",
            (provenance / "evaluator_code_status.txt").read_text() == "",
            (provenance / "evaluator_code_status.txt").read_text(),
        )
        self.require(
            f"{root.name}.clean_evaluator_patch",
            (provenance / "evaluator_code.patch").stat().st_size == 0,
            (provenance / "evaluator_code.patch").stat().st_size,
        )
        return commit

    def _audit_result_identity(
        self,
        *,
        check_prefix: str,
        path: Path,
        payload: dict[str, Any],
        arm: str,
        seed: int,
        action_mode: str,
        window_mode: str,
        checkpoint_hashes: dict[str, str],
    ) -> None:
        expected_arms = {
            "single": {
                "n_players": 1,
                "group_mode": "single",
                "training_group_mode": "single",
            },
            "shared": {
                "n_players": 10,
                "group_mode": "synchronized",
                "training_group_mode": "synchronized",
            },
        }
        identity = f"{check_prefix}.{path.relative_to(self.root)}"
        self.require(f"{identity}.arm", arm in expected_arms, arm)
        expected = expected_arms[arm]
        fields = {
            "dataset_backend": "counterstrike1k",
            "split": "test",
            "map_slug": "dust2",
            "deterministic": True,
            "seed": seed,
            "n_players": expected["n_players"],
            "group_mode": expected["group_mode"],
            "training_group_mode": expected["training_group_mode"],
            "action_mode": action_mode,
            "window_mode": window_mode,
            "checkpoint_sha256": checkpoint_hashes[arm],
        }
        for field_name, expected_value in fields.items():
            self.require(
                f"{identity}.{field_name}",
                payload.get(field_name) == expected_value,
                {"expected": expected_value, "observed": payload.get(field_name)},
            )

    def audit_primary_evaluation(self, checkpoint_hashes: dict[str, str]) -> dict[str, Any]:
        root = self.root / "evaluation" / "test_seed_sweep"
        evaluator_commit = self._audit_eval_provenance(root)
        summary = _read_json(root / "summary.json")
        contract = summary["contract"]
        expected = {
            "split": "test",
            "map_slug": "dust2",
            "dino_model": "dinov2_vitb14",
            "window_mode": "midpoint",
            "deterministic": True,
            "seeds": EXPECTED_PRIMARY_SEEDS,
            "validation_raw_pov_rows_per_seed": 520,
            "metrics_raw_pov_rows_per_seed": 520,
            "single_checkpoint_sha256": checkpoint_hashes["single"],
            "shared_checkpoint_sha256": checkpoint_hashes["shared"],
        }
        for key, value in expected.items():
            self.require(f"primary_eval.{key}", contract.get(key) == value, contract.get(key))
        result_paths = sorted(root.glob("seed_*/*.json"))
        self.require("primary_eval.result_files", len(result_paths) == 6, len(result_paths))
        for path in result_paths:
            payload = _read_json(path)
            _finite_results(payload, path)
            arm = path.stem
            seed = int(path.parent.name.removeprefix("seed_"))
            self.require(
                f"primary_eval.{path.parent.name}.{path.stem}.seed_membership",
                seed in EXPECTED_PRIMARY_SEEDS,
                seed,
            )
            self._audit_result_identity(
                check_prefix="primary_eval",
                path=path,
                payload=payload,
                arm=arm,
                seed=seed,
                action_mode="true",
                window_mode="midpoint",
                checkpoint_hashes=checkpoint_hashes,
            )
            self.require(
                f"primary_eval.{path.parent.name}.{path.stem}.raw_rows",
                payload["validation"]["total_raw_pov_rows"]
                == payload["metrics"]["total_raw_pov_rows"]
                == 520,
                {
                    "validation": payload["validation"]["total_raw_pov_rows"],
                    "metrics": payload["metrics"]["total_raw_pov_rows"],
                },
            )
        return {"evaluator_commit": evaluator_commit, "summary": str(root / "summary.json")}

    def audit_action_evaluation(
        self,
        checkpoint_hashes: dict[str, str],
        *,
        root_name: str = "action_loss_seed_sweep",
        check_prefix: str = "action_eval",
        window_mode: str = "midpoint",
    ) -> dict[str, Any]:
        root = self.root / "evaluation" / root_name
        evaluator_commit = self._audit_eval_provenance(root)
        summary = _read_json(root / "summary.json")
        contract = summary["contract"]
        expected = {
            "split": "test",
            "map_slug": "dust2",
            "deterministic": True,
            "seeds": EXPECTED_ACTION_SEEDS,
            "action_modes": EXPECTED_ACTION_MODES,
            "window_mode": window_mode,
            "validation_raw_pov_rows_per_arm_per_seed": 520,
            "single_checkpoint_sha256": checkpoint_hashes["single"],
            "shared_checkpoint_sha256": checkpoint_hashes["shared"],
        }
        for key, value in expected.items():
            self.require(f"{check_prefix}.{key}", contract.get(key) == value, contract.get(key))
        result_paths = sorted(root.glob("seed_*/*/*.json"))
        self.require(f"{check_prefix}.result_files", len(result_paths) == 40, len(result_paths))
        for path in result_paths:
            payload = _read_json(path)
            _finite_results(payload, path)
            arm = path.parent.name
            seed = int(path.parent.parent.name.removeprefix("seed_"))
            action_mode = path.stem
            self.require(
                f"{check_prefix}.{path.parent.parent.name}.{arm}.{action_mode}.seed_membership",
                seed in EXPECTED_ACTION_SEEDS,
                seed,
            )
            self.require(
                f"{check_prefix}.{path.parent.parent.name}.{arm}.{action_mode}.mode_membership",
                action_mode in EXPECTED_ACTION_MODES,
                action_mode,
            )
            self._audit_result_identity(
                check_prefix=check_prefix,
                path=path,
                payload=payload,
                arm=arm,
                seed=seed,
                action_mode=action_mode,
                window_mode=window_mode,
                checkpoint_hashes=checkpoint_hashes,
            )
            self.require(
                f"{check_prefix}.{path.parent.parent.name}.{path.parent.name}.{path.stem}.raw_rows",
                payload["validation"]["total_raw_pov_rows"] == 520,
                payload["validation"]["total_raw_pov_rows"],
            )
            self.require(
                f"{check_prefix}.{path.parent.parent.name}.{path.parent.name}.{path.stem}.window_mode",
                payload["window_mode"] == window_mode,
                payload["window_mode"],
            )
        return {"evaluator_commit": evaluator_commit, "summary": str(root / "summary.json")}

    def audit_watcher_status(self) -> None:
        expected = {
            "post_pipeline_status.tsv": ("evaluation", "complete"),
            "action_ablation_status.tsv": ("action_evaluation", "complete"),
            "death_action_ablation_status.tsv": ("death_action_evaluation", "complete"),
        }
        for filename, terminal in expected.items():
            lines = (self.root / filename).read_text().splitlines()
            fields = lines[-1].split("\t")
            observed = (fields[1], fields[2])
            self.require(f"watcher.{filename}", observed == terminal, observed)

    def run(self) -> dict[str, Any]:
        self.audit_dataset()
        training_commit = self.audit_code_provenance()
        times = self.audit_pipeline_status()
        frames = self.audit_configs()
        codec = self.audit_training_stage("codec", 1.0, 64)
        single = self.audit_training_stage("single", 5.5, frames["single"])
        shared = self.audit_training_stage("shared", 5.5, frames["shared"])
        telemetry = self.audit_gpu_telemetry(times)
        checkpoint_hashes = {
            "single": single["checkpoint_sha256"],
            "shared": shared["checkpoint_sha256"],
        }
        primary = self.audit_primary_evaluation(checkpoint_hashes)
        action = self.audit_action_evaluation(checkpoint_hashes)
        death_action = self.audit_action_evaluation(
            checkpoint_hashes,
            root_name="death_action_loss_seed_sweep",
            check_prefix="death_action_eval",
            window_mode="first-death",
        )
        self.audit_watcher_status()
        self.require(
            "evaluation.same_commit",
            primary["evaluator_commit"] == action["evaluator_commit"] == death_action["evaluator_commit"],
            {
                "primary": primary["evaluator_commit"],
                "action": action["evaluator_commit"],
                "death_action": death_action["evaluator_commit"],
            },
        )
        return {
            "status": "pass",
            "run_root": str(self.root.resolve()),
            "training_commit": training_commit,
            "evaluator_commit": primary["evaluator_commit"],
            "dataset_selection_sha256": EXPECTED_SELECTION_SHA256,
            "stages": {"codec": codec, "single": single, "shared": shared},
            "gpu_telemetry": telemetry,
            "primary_evaluation": primary,
            "action_evaluation": action,
            "death_action_evaluation": death_action,
            "checks": self.checks,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--expected-training-commit")
    parser.add_argument("--expected-evaluator-commit")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = Auditor(
        args.run_root,
        expected_training_commit=args.expected_training_commit,
        expected_evaluator_commit=args.expected_evaluator_commit,
    ).run()
    output = args.output or args.run_root / "audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(output)


if __name__ == "__main__":
    main()
