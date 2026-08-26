"""Regex-based parsers for the plain-text logs QNAS writes into an experiment
directory. Each parser is incremental: callers pass the previous byte offset
and get back new records plus the offset to resume from, so a live-tailing
dashboard doesn't have to re-read multi-hour-old log files on every refresh.
"""

import re
from pathlib import Path

_GEN_BLOCK_RE = re.compile(
    r"- Generation:\s*(?P<generation>\d+)\s*\n"
    r"- Stage:\s*(?P<stage>\d+)\s*\(gen_start=(?P<gen_start>\d+),\s*"
    r"num_nodes=(?P<num_nodes>\d+),\s*num_ops=(?P<num_ops>\d+)\)\s*\n"
    r"- New architectures discovered:\s*(?P<discovered>\d+)\s*\n"
    r"- Best so far:\s*\[(?P<best_gen>\d+),\s*(?P<best_idx>\d+)\]\s*-->\s*(?P<best_fitness>[\d.]+)\s*\n"
    r"- Fitnesses:\s*\[(?P<fitnesses>[^\]]*)\]",
    re.MULTILINE,
)

_RETRAIN_EPOCH_RE = re.compile(
    r"Experiment:\s*(?P<experiment>\S+)\s*-\s*Epoch\s*\[(?P<epoch>\d+)/(?P<max_epoch>\d+)\]\s*-\s*"
    r"Training loss:\s*(?P<train_loss>[\d.]+)\s*-\s*Validation loss:\s*(?P<val_loss>[\d.]+)\s*-\s*"
    r"Validation accuracy:\s*(?P<val_acc>[\d.]+)%"
)

_RETRAIN_TEST_RE = re.compile(
    r"Experiment:\s*(?P<experiment>\S+)\s*-\s*Test loss:\s*(?P<test_loss>[\d.]+)\s*-\s*"
    r"Test accuracy:\s*(?P<test_acc>[\d.]+)%"
)

_TRAIN_INDIVIDUAL_RE = re.compile(
    r"Individual\s*(?P<individual>\d+)\s*on thread\s*(?P<thread>\d+):\s*"
    r"Best Metric=(?P<metric>[\d.]+),\s*Params=(?P<params>[\d.]+)M,\s*"
    r"Inference Time=(?P<inference_us>[\d.]+)uS"
)

_TIMESTAMP_RE = re.compile(r"^\S+:\s*\S+:\s*(?P<ts>\d{4}-\d{2}-\d{2} [\d:.]+)")


def _read_new_text(path, offset):
    p = Path(path)
    if not p.exists():
        return "", offset
    size = p.stat().st_size
    if size < offset:
        offset = 0  # file was truncated/rotated, start over
    with open(p, "r", errors="replace") as f:
        f.seek(offset)
        text = f.read()
    return text, size


def _floats(csv_text):
    return [float(x) for x in csv_text.split()]


def parse_qnas_generations(path, offset=0):
    """Parse new generation summary blocks from log_QNAS.txt."""
    text, new_offset = _read_new_text(path, offset)
    records = []
    for m in _GEN_BLOCK_RE.finditer(text):
        records.append({
            "generation": int(m.group("generation")),
            "stage": int(m.group("stage")),
            "gen_start": int(m.group("gen_start")),
            "num_nodes": int(m.group("num_nodes")),
            "num_ops": int(m.group("num_ops")),
            "discovered": int(m.group("discovered")),
            "best_fitness": float(m.group("best_fitness")),
            "fitnesses": _floats(m.group("fitnesses")),
        })
    # Re-read from the start of the last incomplete block next time, in case
    # the block was cut mid-write.
    if records:
        last_match_end = list(_GEN_BLOCK_RE.finditer(text))[-1].end()
        new_offset = offset + len(text[:last_match_end].encode("utf-8", errors="replace"))
    return records, new_offset


def parse_retrain_epochs(path, offset=0):
    """Parse new epoch/test lines from retrain.log."""
    text, new_offset = _read_new_text(path, offset)
    epochs = []
    tests = []
    for m in _RETRAIN_EPOCH_RE.finditer(text):
        epochs.append({
            "experiment": m.group("experiment"),
            "epoch": int(m.group("epoch")),
            "max_epoch": int(m.group("max_epoch")),
            "train_loss": float(m.group("train_loss")),
            "val_loss": float(m.group("val_loss")),
            "val_acc": float(m.group("val_acc")),
        })
    for m in _RETRAIN_TEST_RE.finditer(text):
        tests.append({
            "experiment": m.group("experiment"),
            "test_loss": float(m.group("test_loss")),
            "test_acc": float(m.group("test_acc")),
        })
    return epochs, tests, new_offset


def parse_train_individuals(path, offset=0, max_records=200):
    """Parse new per-individual training result lines from train.log."""
    text, new_offset = _read_new_text(path, offset)
    records = []
    for m in _TRAIN_INDIVIDUAL_RE.finditer(text):
        records.append({
            "individual": int(m.group("individual")),
            "thread": int(m.group("thread")),
            "metric": float(m.group("metric")),
            "params_m": float(m.group("params")),
            "inference_us": float(m.group("inference_us")),
        })
    if len(records) > max_records:
        records = records[-max_records:]
    return records, new_offset
