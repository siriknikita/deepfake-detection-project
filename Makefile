# Hyperplane-Forge — developer commands
#
# All Python commands route through `uv run` so the project's virtual
# environment is set up and used implicitly. Rust commands use cargo.
# `make help` lists the available targets.

.PHONY: help install dev-install build build-release fmt fmt-check lint typecheck \
        test test-rust test-python paper paper-watch clean pre-commit

UV ?= uv
CARGO ?= cargo
TYPST ?= typst

help:  ## List available targets
	@awk 'BEGIN{FS=":.*##"; printf "Targets:\n"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[1m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## Install runtime deps in a uv-managed venv
	$(UV) sync

dev-install:  ## Install dev deps and build the Rust extension in-place
	$(UV) sync --extra dev
	$(UV) run maturin develop --release

build:  ## Build the Rust extension (debug)
	$(UV) run maturin develop

build-release:  ## Build the Rust extension (release; LTO + native CPU)
	$(UV) run maturin develop --release

fmt:  ## Format Rust and Python code in-place
	$(CARGO) fmt --all
	$(UV) run ruff format python tests

fmt-check:  ## Check formatting without modifying files
	$(CARGO) fmt --all -- --check
	$(UV) run ruff format --check python tests

lint:  ## Lint Rust (clippy) and Python (ruff)
	$(CARGO) clippy --all-targets --all-features -- -D warnings
	$(UV) run ruff check python tests

typecheck:  ## Run mypy in strict mode on Python sources
	$(UV) run mypy

test: test-rust test-python  ## Run the full test suite

test-rust:  ## Cargo unit tests (in-crate)
	$(CARGO) test --release

test-python:  ## Pytest suite (requires `make build-release` first)
	$(UV) run pytest

paper:  ## Compile the diploma paper to PDF
	$(TYPST) compile paper/main.typ paper/main.pdf

paper-watch:  ## Recompile the paper on file changes
	$(TYPST) watch paper/main.typ paper/main.pdf

pre-commit:  ## Run all pre-commit hooks against every file
	$(UV) run pre-commit run --all-files

clean:  ## Remove build artifacts and caches
	$(CARGO) clean
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -f paper/main.pdf
