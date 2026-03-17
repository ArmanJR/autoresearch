EXPERIMENT ?=
LDFLAGS := -X main.experimentName=$(EXPERIMENT)

# Detect target platform (default: build for Jetson)
GOOS ?= linux
GOARCH ?= arm64

# Find which director dir contains this experiment's config.
# Searches director*/configs/<EXPERIMENT>.json and returns the parent dir.
DIRECTOR_DIR := $(firstword $(patsubst %/configs/$(EXPERIMENT).json,%,$(wildcard director*/configs/$(EXPERIMENT).json)))

.PHONY: director deploy list help

help:
	@echo "Usage:"
	@echo "  make director EXPERIMENT=mad-scientist-3-16  Build director for an experiment"
	@echo "  make deploy   EXPERIMENT=mad-scientist-3-16  Build + copy binary into experiment dir"
	@echo "  make list                                    List available experiment configs"
	@echo ""
	@echo "Options:"
	@echo "  GOOS=darwin GOARCH=arm64    Build for macOS Apple Silicon (default: linux/arm64)"

# Validate that EXPERIMENT is set and config exists in some director dir
check-experiment:
ifndef EXPERIMENT
	$(error EXPERIMENT is required. Run 'make list' to see available configs)
endif
ifeq ($(DIRECTOR_DIR),)
	$(error config not found for experiment '$(EXPERIMENT)' in any director*/configs/)
endif

# Build the director binary for the given experiment
director: check-experiment
	cd $(DIRECTOR_DIR) && \
		GOOS=$(GOOS) GOARCH=$(GOARCH) go build \
		-ldflags '$(LDFLAGS)' \
		-o ../$(EXPERIMENT)/director .
	@echo "built $(EXPERIMENT)/director ($(GOOS)/$(GOARCH)) [source: $(DIRECTOR_DIR)]"

# Build + copy .env into experiment dir
deploy: director
	@test -f $(DIRECTOR_DIR)/.env && \
		cp $(DIRECTOR_DIR)/.env $(EXPERIMENT)/.env || \
		echo "warning: no $(DIRECTOR_DIR)/.env found, skipping .env copy"
	@echo "deployed director to $(EXPERIMENT)/"

# List available experiment configs across all director dirs
list:
	@echo "Available experiments:"
	@for f in director*/configs/*.json; do \
		dir=$$(echo "$$f" | cut -d/ -f1); \
		name=$$(basename "$$f" .json); \
		echo "  $$name  ($$dir)"; \
	done
