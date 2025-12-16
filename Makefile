PYTHON ?= python3
VENV ?= .venv
REQS ?= requirements.txt
PIP ?= $(VENV)/bin/pip

.PHONY: venv clean

venv: $(PIP)
	$(PIP) install -r $(REQS)

$(PIP):
	@echo "Creating virtual environment in $(VENV) using virtualenv..."
	@$(PYTHON) -m pip install --user --upgrade virtualenv
	@$(PYTHON) -m virtualenv $(VENV)
	@if [ ! -x "$(PIP)" ] && [ -x "$(VENV)/bin/pip3" ]; then ln -s pip3 "$(PIP)"; fi
	@if [ ! -x "$(PIP)" ]; then echo "pip not found in $(VENV)"; exit 1; fi
	@$(PIP) install --upgrade pip

clean:
	rm -rf $(VENV)
