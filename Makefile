.PHONY: test lint format lint-fix smoke smoke-synthid bootstrap-synthid docker-synthid-build docker-synthid-help \
	smoke-ctrlregen bootstrap-ctrlregen docker-ctrlregen-build docker-ctrlregen-help \
	smoke-markllm bootstrap-markllm docker-markllm-build docker-markllm-help \
	smoke-markdiffusion bootstrap-markdiffusion docker-markdiffusion-build docker-markdiffusion-help \
	bench-synthid-text bench-full bench-semantic \
	docker-core-build docker-core-help serve compose-up compose-up-heavy compose-check \
	install-skill install-claude-code-skill install-claude-code-text-skill \
	install-claude-project-skill package-cowork-skill package-cowork-text-skill \
	install-cursor-text-skill plugin-validate clean

SCRIPTS := service/scripts
PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)

# Rewrite backend defaults: DeepSeek (cross-model, non-origin). Override with
# e.g. `make REWRITE_MODEL=<m> REWRITE_BASE_URL=http://127.0.0.1:8000 REWRITE_ALLOW_REMOTE= bench-full`.
REWRITE_BACKEND ?= openai-compatible
REWRITE_MODEL ?= deepseek-v4-flash
REWRITE_BASE_URL ?= https://api.deepseek.com
REWRITE_ALLOW_REMOTE ?= --rewrite-allow-remote

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check service tests

format:
	$(PYTHON) -m ruff format --check service tests

lint-fix:
	$(PYTHON) -m ruff check --fix service tests

smoke:
	./service/scripts/smoke.sh

smoke-synthid:
	./service/scripts/smoke_synthid.sh

bootstrap-synthid:
	./service/scripts/setup_reverse_synthid.sh

docker-synthid-build:
	docker build -f service/Dockerfile.synthid -t watermarks-remover-synthid service/

docker-synthid-help:
	docker run --rm watermarks-remover-synthid --help

smoke-ctrlregen:
	./service/scripts/smoke_ctrlregen.sh

bootstrap-ctrlregen:
	./service/scripts/setup_ctrlregen.sh

docker-ctrlregen-build:
	docker build -f service/Dockerfile.ctrlregen -t watermarks-remover-ctrlregen service/

docker-ctrlregen-help:
	docker run --rm watermarks-remover-ctrlregen --help

smoke-markllm:
	./service/scripts/smoke_markllm.sh

bootstrap-markllm:
	./service/scripts/setup_markllm.sh

docker-markllm-build:
	docker build -f service/Dockerfile.markllm -t watermarks-remover-markllm service/

docker-markllm-help:
	docker run --rm watermarks-remover-markllm --help

smoke-markdiffusion:
	./service/scripts/smoke_markdiffusion.sh

bench-synthid-text:
	@if [ -z "$(MARKLLM_DIR)" ]; then \
	  echo "bench-synthid-text skipped (set MARKLLM_DIR; see docs/synthid-text-benchmark.md)"; \
	else \
	  echo "run: $(PYTHON) $(SCRIPTS)/bench_synthid_text.py --markllm-dir $(MARKLLM_DIR) --rewrite-model <model> --rewrite-backend <backend>"; \
	fi

# Full strategy-search benchmark: all tactics, robust margin, semantic axis
# required. Use a non-watermarked rewrite backend.
bench-full:
	@if [ -z "$(MARKLLM_DIR)" ]; then \
	  echo "bench-full skipped (set MARKLLM_DIR; see docs/synthid-text-benchmark.md)"; \
	elif [ -z "$(REWRITE_MODEL)" ] || [ -z "$(REWRITE_BACKEND)" ]; then \
	  echo "error: REWRITE_MODEL and REWRITE_BACKEND must be set (defaults: deepseek-v4-flash / openai-compatible)"; exit 1; \
	else \
	  "$(MARKLLM_DIR)/.venv/bin/python" $(SCRIPTS)/bench_synthid_text.py --markllm-dir $(MARKLLM_DIR) --corpus benchmarks/corpus-large --docs 20 --seeds 3 --max-new-tokens 300 --target-margin 0.03 --require-semantic --mode strategy --coverage-floor 0.5 --eval-split 0.8 --humanize-intensity 0.4 --rewrite-backend $(REWRITE_BACKEND) --rewrite-model $(REWRITE_MODEL) --rewrite-base-url $(REWRITE_BASE_URL) $(REWRITE_ALLOW_REMOTE); \
	fi

# Install the semantic-divergence dependency into the MarkLLM venv and run with a
# writable HF cache, so the sem-div column is never a silent '—'.
bench-semantic:
	@if [ -z "$(MARKLLM_DIR)" ]; then \
	  echo "bench-semantic skipped (set MARKLLM_DIR)"; \
	elif [ -z "$(REWRITE_MODEL)" ] || [ -z "$(REWRITE_BACKEND)" ]; then \
	  echo "error: REWRITE_MODEL and REWRITE_BACKEND must be set (defaults: deepseek-v4-flash / openai-compatible)"; exit 1; \
	else \
	  mkdir -p $(CURDIR)/.hf-cache && \
	  "$(MARKLLM_DIR)/.venv/bin/python" -m pip install -r $(SCRIPTS)/requirements-semantic.txt && \
	  HF_HOME=$(CURDIR)/.hf-cache "$(MARKLLM_DIR)/.venv/bin/python" $(SCRIPTS)/bench_synthid_text.py --markllm-dir $(MARKLLM_DIR) --require-semantic --rewrite-backend $(REWRITE_BACKEND) --rewrite-model $(REWRITE_MODEL) --rewrite-base-url $(REWRITE_BASE_URL) $(REWRITE_ALLOW_REMOTE); \
	fi

bootstrap-markdiffusion:
	./service/scripts/setup_markdiffusion.sh

docker-markdiffusion-build:
	docker build -f service/Dockerfile.markdiffusion -t watermarks-remover-markdiffusion service/

docker-markdiffusion-help:
	docker run --rm watermarks-remover-markdiffusion --help

# Core HTTP service (text + file/image metadata cleaning).
docker-core-build:
	docker build -f service/Dockerfile -t watermarks-remover service/

docker-core-help:
	docker run --rm watermarks-remover /app/scripts/server.py --help

# Run the HTTP service locally (stdlib only, no Docker).
serve:
	$(PYTHON) $(SCRIPTS)/server.py --host 127.0.0.1 --port 8765

compose-up:
	docker compose up --build -d

compose-up-heavy:
	docker compose --profile harness --profile heavy up --build -d

compose-check:
	./compose-check.sh

# Grok (symlink; edits in this checkout are picked up live).
install-skill:
	mkdir -p $(HOME)/.grok/skills
	ln -sfn $(CURDIR)/skills/remove-ai-marks $(HOME)/.grok/skills/remove-ai-marks
	@echo "linked -> $(HOME)/.grok/skills/remove-ai-marks"

# Claude Code: personal skills (~/.claude/skills), available in every project.
install-claude-code-skill:
	$(PYTHON) install_skill.py --skill remove-ai-marks --target claude-code

install-claude-code-text-skill:
	$(PYTHON) install_skill.py --skill clean-user-facing-text --target claude-code

# Claude Code: project skills (<PROJECT>/.claude/skills), shared via the repo.
install-claude-project-skill:
	$(PYTHON) install_skill.py --skill remove-ai-marks --target claude-project \
	  --project-dir $(or $(PROJECT),$(CURDIR))

# Cowork / cloud sessions load skills enabled for the claude.ai account, so
# build an upload bundle for Customize > Skills instead of writing to a dir.
package-cowork-skill:
	$(PYTHON) install_skill.py --skill remove-ai-marks --target cowork --force

package-cowork-text-skill:
	$(PYTHON) install_skill.py --skill clean-user-facing-text --target cowork --force

install-cursor-text-skill:
	$(PYTHON) install_skill.py --skill clean-user-facing-text --target cursor

# Claude Code plugin + single-plugin marketplace manifests (.claude-plugin/).
# Needs the claude CLI; tests/test_plugin_manifest.py covers the same files
# structurally so CI stays CLI-free.
plugin-validate:
	claude plugin validate . --strict

clean:
	rm -rf dist
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .venv
