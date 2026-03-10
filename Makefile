EXPERIMENT ?=
DIRECTOR_DIR := director
LDFLAGS := -X main.experimentName=$(EXPERIMENT)

# Detect target platform (default: build for Jetson)
GOOS ?= linux
GOARCH ?= arm64

.PHONY: director deploy list help

help:
	@echo "Usage:"
	@echo "  make director EXPERIMENT=mad-scientist    Build director for an experiment"
	@echo "  make deploy   EXPERIMENT=mad-scientist    Build + copy binary into experiment dir"
	@echo "  make list                                 List available experiment configs"
	@echo ""
	@echo "Options:"
	@echo "  GOOS=darwin GOARCH=arm64    Build for macOS Apple Silicon (default: linux/arm64)"

# Validate that EXPERIMENT is set and config exists
check-experiment:
ifndef EXPERIMENT
	$(error EXPERIMENT is required. Run 'make list' to see available configs)
endif
	@test -f $(DIRECTOR_DIR)/configs/$(EXPERIMENT).json || \
		(echo "error: config not found: $(DIRECTOR_DIR)/configs/$(EXPERIMENT).json" && exit 1)

# Build the director binary for the given experiment
director: check-experiment
	cd $(DIRECTOR_DIR) && \
		GOOS=$(GOOS) GOARCH=$(GOARCH) go build \
		-ldflags '$(LDFLAGS)' \
		-o ../$(EXPERIMENT)/director .
	@echo "built $(EXPERIMENT)/director ($(GOOS)/$(GOARCH))"

# Build + copy .env into experiment dir
deploy: director
	@test -f $(DIRECTOR_DIR)/.env && \
		cp $(DIRECTOR_DIR)/.env $(EXPERIMENT)/.env || \
		echo "warning: no $(DIRECTOR_DIR)/.env found, skipping .env copy"
	@echo "deployed director to $(EXPERIMENT)/"

# List available experiment configs
list:
	@echo "Available experiments:"
	@ls $(DIRECTOR_DIR)/configs/*.json 2>/dev/null | \
		sed 's|.*/||; s|\.json$$||' | \
		while read name; do echo "  $$name"; done
