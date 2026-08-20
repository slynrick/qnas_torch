import shlex
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def build_argv(mode, config_path, experiment_path, extra_args):
    extra = shlex.split(extra_args) if extra_args else []

    if mode == "evolve":
        return [
            "uv", "run", "python", str(SRC_DIR / "run_evolution.py"),
            "--experiment_path", experiment_path,
            "--config_file", config_path,
        ] + extra

    if mode == "retrain":
        # config_path is stored for provenance/listing only - retrain_model.py
        # reads log_params_evolution.txt from experiment_path, not a config file.
        return [
            "uv", "run", "python", str(SRC_DIR / "retrain_model.py"),
            "--experiment_path", experiment_path,
        ] + extra

    if mode == "pipeline":
        return [
            "bash", str(SCRIPTS_DIR / "run_pipeline.sh"),
            "-e", experiment_path,
            "-c", config_path,
        ] + extra

    raise ValueError(f"Unknown mode: {mode}")
