#!/bin/bash
# Syncs only the experiment results (experiment_*/), the config files (configs/), the
# .runs.txt scratch file, and optionally the job queue state (.qnas_queue/) between this
# checkout and a copy of the project on a remote machine, using rsync over ssh.
# Code, .venv, datasets and everything else are left out.
#
# Usage:
#   scripts/sync-experiments.sh [options] <ssh_host> <remote_project_path>
#
#   scripts/sync-experiments.sh puc_dualgpu1 '~/workspace/qnas_torch'        # pull (default)
#   scripts/sync-experiments.sh -n puc_dualgpu1 '~/workspace/qnas_torch'     # dry run
#   scripts/sync-experiments.sh --push puc_dell /home/user/qnas_torch        # push local -> remote
#   scripts/sync-experiments.sh --queue puc_dualgpu1 '~/workspace/qnas_torch'  # also pull .qnas_queue/
#
# Arguments:
#   ssh_host             a Host alias defined in ~/.ssh/config
#   remote_project_path  project root on the remote (the dir holding configs/ and experiment_*/).
#                        Quote it ('~/...') to have ~ expanded on the remote side.
#
# Options:
#   -n, --dry-run        show what would be transferred without copying anything
#   --push               copy local -> remote instead of remote -> local
#   --experiments-only   skip configs/
#   --configs-only       skip experiment_*/
#   --queue              also sync .qnas_queue/ (queue.db, logs/, configs/ snapshots);
#                        worker.pid is always skipped since it names a process on one
#                        specific host and is meaningless copied elsewhere
#   --exclude PATTERN    extra rsync exclude pattern (repeatable), e.g. --exclude '*.pth'
#   -h, --help           show this help
#
# Safety: files are never deleted on the receiving side, and a file that is newer on the
# receiving side is never overwritten (rsync --update).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/path_config.sh"

usage() { sed -n '2,/^$/{s/^# \{0,1\}//;p}' "${BASH_SOURCE[0]}"; }

direction="pull"
dry_run=()
sync_experiments=true
sync_configs=true
sync_queue=false
extra_excludes=()
positional=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--dry-run) dry_run=(--dry-run); shift ;;
        --push) direction="push"; shift ;;
        --pull) direction="pull"; shift ;;
        --experiments-only) sync_configs=false; shift ;;
        --configs-only) sync_experiments=false; shift ;;
        --queue) sync_queue=true; shift ;;
        --exclude) extra_excludes+=(--exclude "$2"); shift 2 ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "error: unknown option $1" >&2; usage >&2; exit 1 ;;
        *) positional+=("$1"); shift ;;
    esac
done

if [[ ${#positional[@]} -ne 2 ]]; then
    echo "error: expected <ssh_host> <remote_project_path>" >&2
    usage >&2
    exit 1
fi
host="${positional[0]}"
remote_path="${positional[1]%/}"

if ! $sync_experiments && ! $sync_configs; then
    echo "error: --experiments-only and --configs-only are mutually exclusive" >&2
    exit 1
fi

# The host must be an alias from the ssh config (Host lines, including files pulled in by Include).
ssh_config="${HOME}/.ssh/config"
ssh_config_files=("$ssh_config")
[[ -d "${HOME}/.ssh/config.d" ]] && ssh_config_files+=("${HOME}/.ssh/config.d")
if ! ssh -G "$host" >/dev/null 2>&1 \
        || ! grep -rhiE '^\s*Host\s' "${ssh_config_files[@]}" 2>/dev/null \
            | tr -s ' \t' '\n' | grep -qxF "$host"; then
    echo "error: host '$host' not found in ${ssh_config}" >&2
    echo "known hosts:" >&2
    grep -rhiE '^\s*Host\s' "${ssh_config_files[@]}" 2>/dev/null \
        | sed -E 's/^\s*Host\s+//I' | tr ' ' '\n' | grep -v '[*?]' | sed 's/^/  /' >&2
    exit 1
fi

if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" "test -d ${remote_path}/configs" 2>/dev/null; then
    echo "error: ${host}:${remote_path} is unreachable or has no configs/ dir (is it the project root?)" >&2
    exit 1
fi

# Whitelist: descend into the top-level dirs we want, take everything under them, drop the rest.
# rsync applies the first matching rule, so the extra excludes go before the includes.
filters=(--exclude '__pycache__/' --exclude '*.pyc' --exclude '/.qnas_queue/worker.pid' "${extra_excludes[@]}")
$sync_experiments && filters+=(--include '/experiment_*/***')
$sync_configs && filters+=(--include '/configs/***')
$sync_queue && filters+=(--include '/.qnas_queue/***')
filters+=(--include '/.runs.txt')
filters+=(--exclude '*')

if [[ "$direction" == "pull" ]]; then
    src="${host}:${remote_path}/"
    dst="${PROJECT_DIR}/"
else
    src="${PROJECT_DIR}/"
    dst="${host}:${remote_path}/"
fi

echo "sync (${direction}): ${src} -> ${dst}"
rsync -a --update --human-readable --info=stats1,progress2 --partial \
    "${dry_run[@]}" "${filters[@]}" "$src" "$dst"
