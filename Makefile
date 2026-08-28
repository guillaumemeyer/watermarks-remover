.PHONY: test lint format lint-fix smoke smoke-synthid bootstrap-synthid docker-synthid-build docker-synthid-help \
	smoke-ctrlregen bootstrap-ctrlregen docker-ctrlregen-build docker-ctrlregen-help \
	smoke-markllm bootstrap-markllm docker-markllm-build docker-markllm-help \
	smoke-markdiffusion bootstrap-markdiffusion docker-markdiffusion-build docker-markdiffusion-help \
	bench-synthid-text \
	docker-core-build docker-core-help serve compose-up compose-up-heavy compose-check \
	install-skill install-claude-code-skill install-claude-code-text-skill \
	install-claude-project-skill package-cowork-skill package-cowork-text-skill \
	install-cursor-text-skill plugin-validate clean

SCRIPTS := service/scripts
PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check service tests

format:
	$(PYTHON) -m ruff format --check service tests

lint-fix:
	$(PYTHON) -m ruff check --fix service tests

smoke:
	-python3 $(SCRIPTS)/inspect_text.py tests/fixtures/sample_watermarked.txt
	python3 $(SCRIPTS)/clean_text.py tests/fixtures/sample_watermarked.txt -o /tmp/wm.cleaned.txt --stats
	python3 $(SCRIPTS)/rewrite_text.py tests/fixtures/sample_watermarked.txt --backend print-prompt >/dev/null
	-python3 $(SCRIPTS)/inspect_file.py tests/fixtures/sample_ai.md
	python3 $(SCRIPTS)/clean_file.py tests/fixtures/sample_ai.md -o /tmp/sample_ai.cleaned.md
	python3 $(SCRIPTS)/clean_file.py tests/fixtures/sample_ai.html -o /tmp/sample_ai.cleaned.html
	python3 $(SCRIPTS)/clean_file.py tests/fixtures/sample_meta.svg -o /tmp/sample_meta.cleaned.svg
	@echo "smoke ok"

smoke-synthid:
	@if [ -z "$(REVERSE_SYNTHID_DIR)" ]; then \
	  echo "smoke-synthid skipped (set REVERSE_SYNTHID_DIR)"; \
	else \
	  $(PYTHON) $(SCRIPTS)/score_synthid.py --help >/dev/null && echo "score_synthid adapter present"; \
	fi

bootstrap-synthid:
	./service/scripts/setup_synthid.sh

docker-synthid-build:
	docker build -f service/Dockerfile.synthid -t watermarks-remover-synthid-scorer service/

docker-synthid-help:
	docker run --rm watermarks-remover-synthid-scorer --help

smoke-ctrlregen:
	@if [ -z "$(NOAI_WATERMARK_DIR)" ]; then \
	  echo "smoke-ctrlregen skipped (set NOAI_WATERMARK_DIR)"; \
	else \
	  $(PYTHON) $(SCRIPTS)/clean_ctrlregen.py --help >/dev/null && echo "clean_ctrlregen adapter present"; \
	fi

bootstrap-ctrlregen:
	./service/scripts/setup_ctrlregen.sh

docker-ctrlregen-build:
	docker build -f service/Dockerfile.ctrlregen -t watermarks-remover-ctrlregen service/

docker-ctrlregen-help:
	docker run --rm watermarks-remover-ctrlregen --help

smoke-markllm:
	@if [ -z "$(MARKLLM_DIR)" ]; then \
	  echo "smoke-markllm skipped (set MARKLLM_DIR)"; \
	else \
	  $(PYTHON) $(SCRIPTS)/detect_text_watermark.py --help >/dev/null && echo "detect_text_watermark adapter present"; \
	fi

bootstrap-markllm:
	./service/scripts/setup_markllm.sh

docker-markllm-build:
	docker build -f service/Dockerfile.markllm -t watermarks-remover-markllm service/

docker-markllm-help:
	docker run --rm watermarks-remover-markllm --help

smoke-markdiffusion:
	@if [ -z "$(MARKDIFFUSION_DIR)" ]; then \
	  echo "smoke-markdiffusion skipped (set MARKDIFFUSION_DIR)"; \
	else \
	  $(PYTHON) $(SCRIPTS)/markdiffusion_harness.py --help >/dev/null && echo "markdiffusion_harness adapter present"; \
	fi

bench-synthid-text:
	@if [ -z "$(MARKLLM_DIR)" ]; then \
	  echo "bench-synthid-text skipped (set MARKLLM_DIR; see docs/synthid-text-benchmark.md)"; \
	else \
	  echo "run: $(PYTHON) $(SCRIPTS)/bench_synthid_text.py --markllm-dir $(MARKLLM_DIR) --rewrite-model <model> --rewrite-backend <backend>"; \
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
