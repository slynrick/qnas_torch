.DEFAULT_GOAL := help

space := $(subst ,, )
QUEUE := scripts/qnas-queue.sh
PIPELINE := scripts/run_pipeline.sh

# Overridable on the command line, e.g.:
#   make queue-add CONFIG=configs/config_files_cifar/01_deterministic_13-8-4.yml \
#       EXP=experiment_cifar10_progressive/exp8 EXTRA="-d cifar10 -M -T -X"
MODE     ?= pipeline
CONFIG   ?=
EXP      ?=
EXTRA    ?=
PRIORITY ?= 0
GPU_IDS  ?=
WORKERS  ?= 1
GPU_POOL ?=
ID       ?=
DATASET  ?= cifar10
ARGS     ?=
A        ?=
B        ?=
EVOLVE_ARGS ?=
HOST     ?=
REMOTE   ?=
SYNC_ARGS ?=

.PHONY: help sync test test-cov lint lint-fix check \
	queue-add queue-list queue-status queue-start queue-stop \
	queue-logs queue-logs-summary queue-logs-all queue-logs-running queue-watch queue-remove queue-cancel queue-retry \
	queue pipeline diff-runs sync-remote queue-watch-remote clean

help: ## Show this help
	@echo "QNAS-torch project commands"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Common variables: CONFIG, EXP, EXTRA, ID, MODE (default: pipeline), GPU_IDS, WORKERS (default: 1), GPU_POOL"
	@echo "Example: make queue-add CONFIG=configs/config_files_cifar/01_deterministic_13-8-4.yml \\"
	@echo "             EXP=experiment_cifar10_progressive/exp8 EXTRA=\"-d cifar10 -M -T -X\""
	@echo "         make queue-start WORKERS=2 GPU_POOL=\"0,1\"   # workers pick a free pool GPU per job"

sync: ## Install/sync project dependencies with uv
	uv sync

## --- Quality ------------------------------------------------------------

test: ## Run the test suite (CPU only; ARGS="-k name -x" passes extra pytest args)
	uv run pytest $(ARGS)

test-cov: ## Run the test suite with a coverage report for src/
	uv run pytest --cov=src --cov-report=term-missing $(ARGS)

lint: ## Lint with ruff (correctness rules only, see [tool.ruff] in pyproject.toml)
	uv run ruff check .

lint-fix: ## Lint and apply ruff's safe autofixes
	uv run ruff check . --fix

check: lint test ## Lint, then run the tests

## --- Experiment queue (src/qnas_queue) ---------------------------------

# `make queue <sub>` is an alias for `make queue-<sub>` (e.g. `make queue list`).
# The word after `queue` is a goal make would otherwise reject, so it is
# swallowed by an empty phony rule and dispatched from the `queue` recipe.
QUEUE_SUBS := add list status start stop logs logs-summary logs-all logs-running watch remove cancel retry
ifeq ($(firstword $(MAKECMDGOALS)),queue)
QUEUE_SUB := $(word 2,$(MAKECMDGOALS))
ifneq ($(QUEUE_SUB),)
.PHONY: $(QUEUE_SUB)
$(QUEUE_SUB):
	@:
endif
endif

queue: ## Alias: make queue <add|list|status|start|stop|logs|logs-summary|logs-all|logs-running|watch|remove|cancel|retry>
	@case " $(QUEUE_SUBS) " in \
		*" $(QUEUE_SUB) "*) $(MAKE) --no-print-directory queue-$(QUEUE_SUB) ;; \
		*) echo "usage: make queue <$(subst $(space),|,$(QUEUE_SUBS))>" >&2; exit 1 ;; \
	esac

queue-add: ## Queue a job (needs CONFIG=, EXP=; optional MODE=, EXTRA=, PRIORITY=, GPU_IDS=)
	@test -n "$(CONFIG)" || (echo "error: CONFIG=<path to yml> is required" >&2; exit 1)
	@test -n "$(EXP)" || (echo "error: EXP=<experiment_path> is required" >&2; exit 1)
	$(QUEUE) add --mode $(MODE) --config $(CONFIG) --experiment-path $(EXP) \
		--extra "$(EXTRA)" --priority $(PRIORITY) $(if $(GPU_IDS),--gpu-ids "$(GPU_IDS)")

queue-list: ## List queued/running/past jobs
	$(QUEUE) list

queue-status: ## Show worker and queue status
	$(QUEUE) status

queue-start: ## Start background worker(s) (also resumes stopped jobs); WORKERS= concurrency, GPU_POOL= shared GPU pool
	$(QUEUE) start --workers $(WORKERS) $(if $(GPU_POOL),--gpu-pool "$(GPU_POOL)")

queue-stop: ## Stop every worker and the jobs it is running
	$(QUEUE) stop

queue-logs: ## Follow logs of the current (or ID=) job; auto-continues into the next queued job
	$(QUEUE) logs $(ID) -f

queue-logs-summary: ## Follow the evolution-level summary log (log_QNAS.txt) instead of raw output
	$(QUEUE) logs $(ID) -f --summary

queue-logs-all: ## Follow the raw log and the summary log together, prefixed [detail]/[summary]
	$(QUEUE) logs $(ID) -f --both

queue-logs-running: ## Follow every currently running job's log at once, prefixed by job id (WORKERS > 1)
	$(QUEUE) logs -f --all-jobs

queue-watch: ## Live dashboard: every running job's generation/best fitness updating in place, detail logs below
	$(QUEUE) watch

queue-remove: ## Delete a job from the queue/history (needs ID=)
	@test -n "$(ID)" || (echo "error: ID=<job id> is required" >&2; exit 1)
	$(QUEUE) remove $(ID)

queue-cancel: ## Cancel a queued job, kept in history as 'cancelled' (needs ID=)
	@test -n "$(ID)" || (echo "error: ID=<job id> is required" >&2; exit 1)
	$(QUEUE) cancel $(ID)

queue-retry: ## Re-queue a failed/stopped/cancelled job (needs ID=)
	@test -n "$(ID)" || (echo "error: ID=<job id> is required" >&2; exit 1)
	$(QUEUE) retry $(ID)

## --- Direct (foreground, not queued) runs ------------------------------

pipeline: ## Run evolve -> retrain -> infographic directly, blocking (needs EXP=, CONFIG=; optional DATASET=, ARGS=)
	@test -n "$(EXP)" || (echo "error: EXP=<experiment_path> is required" >&2; exit 1)
	@test -n "$(CONFIG)" || (echo "error: CONFIG=<path to yml> is required" >&2; exit 1)
	$(PIPELINE) -e $(EXP) -c $(CONFIG) -d $(DATASET) $(ARGS)

## --- Run comparison ---------------------------------------------------

diff-runs: ## Diff the effective config of two runs/configs (needs A=, B=; EVOLVE_ARGS= flags to render a .yml side)
	@test -n "$(A)" || (echo "error: A=<experiment dir | log_params file | .yml> is required" >&2; exit 1)
	@test -n "$(B)" || (echo "error: B=<experiment dir | log_params file | .yml> is required" >&2; exit 1)
	uv run python src/diff_runs.py $(A) $(B) --evolve-args "$(EVOLVE_ARGS)"

## --- Housekeeping -------------------------------------------------------

sync-remote: ## Pull experiment_*/ and configs/ from a remote copy (needs HOST=<ssh alias> REMOTE=<path>; SYNC_ARGS="--push -n")
	@test -n "$(HOST)" || (echo "error: HOST=<Host alias from ~/.ssh/config> is required" >&2; exit 1)
	@test -n "$(REMOTE)" || (echo "error: REMOTE=<remote project path> is required" >&2; exit 1)
	scripts/sync-experiments.sh $(SYNC_ARGS) $(HOST) '$(REMOTE)'

queue-watch-remote: ## Live queue-watch dashboard of a remote host's own queue, read over ssh (needs HOST=<ssh alias> REMOTE=<path>)
	@test -n "$(HOST)" || (echo "error: HOST=<Host alias from ~/.ssh/config> is required" >&2; exit 1)
	@test -n "$(REMOTE)" || (echo "error: REMOTE=<remote project path> is required" >&2; exit 1)
	ssh -t $(HOST) "cd '$(REMOTE)' && make queue-watch"

clean: ## Remove Python bytecode caches
	find . -type d -name '__pycache__' -not -path './.venv/*' -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -not -path './.venv/*' -delete
