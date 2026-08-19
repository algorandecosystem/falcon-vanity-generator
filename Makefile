# pq-vanity — build & dependency helpers.
#
#   make deps         install the Rust toolchain (and report CUDA/Go status)
#   make build        build the CPU release binary
#   make build-cuda   build the release binary with the CUDA backend (sm_120+sm_121)
#   make test         run the test suite
#   make selftest     validate the CUDA pipeline vs the CPU oracle (needs build-cuda)
#
# Run `make` or `make help` for the full list.

SHELL := /bin/bash

# Parallel build, capped at cores-1 (leave one for the OS); fallback 4.
JOBS  := $(shell nproc --ignore=1 2>/dev/null || echo 4)

# Resolve cargo from PATH, else a rustup install under $HOME/.cargo.
CARGO := $(shell command -v cargo 2>/dev/null || echo $(HOME)/.cargo/bin/cargo)

# CUDA SM targets (override e.g. `make build-cuda ARCHS=121`).
ARCHS ?= 120,121

# Self-documenting hits dir / selftest size defaults.
OUT    ?= ./hits
ITEMS  ?= 200000

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@echo "pq-vanity targets:"
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
		| sort | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Vars: JOBS=$(JOBS)  ARCHS=$(ARCHS)  CARGO=$(CARGO)"

.PHONY: deps
deps: ## Install build dependencies (Rust toolchain; reports CUDA/Go)
	@./scripts/install-deps.sh

.PHONY: build
build: ## Build the CPU release binary (target/release/pq-vanity)
	$(CARGO) build --release -j $(JOBS)

.PHONY: build-cuda
build-cuda: ## Build the release binary with the CUDA backend (needs nvcc)
	PQ_CUDA_ARCHS=$(ARCHS) $(CARGO) build -p pq-vanity --release --features cuda -j $(JOBS)

.PHONY: patch-cuda-rsqrt
patch-cuda-rsqrt: ## Fix CUDA<=13.1 / glibc>=2.41 rsqrt header clash (needs CUDA write access)
	@./scripts/patch-cuda-rsqrt.sh

.PHONY: test
test: ## Run the test suite (KATs, formats, fast-path shim)
	$(CARGO) test --release -j $(JOBS)

.PHONY: selftest
selftest: ## Validate the CUDA device pipeline vs the CPU oracle (needs build-cuda)
	./target/release/pq-vanity gpu-selftest --items $(ITEMS)

.PHONY: derive
derive: ## Sanity check: derive the KAT address for entropy byte 0
	$(CARGO) run --release -q -- derive --entropy-byte 0

.PHONY: fmt
fmt: ## Format the Rust sources
	$(CARGO) fmt

.PHONY: clean
clean: ## Remove build artifacts
	$(CARGO) clean
