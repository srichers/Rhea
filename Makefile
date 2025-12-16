PYTHON ?= python3
VENV ?= .venv
REQS ?= requirements.txt
PIP ?= $(VENV)/bin/pip

.PHONY: venv clean

venv: $(PIP)
	$(PIP) install -r $(REQS)

$(PIP):
	@echo "Creating virtual environment in $(VENV)..."
	@# Try stdlib venv (with and without ensurepip), then fall back to virtualenv installed via system pip
	@set -e; \
	if $(PYTHON) -m venv $(VENV) 2>/dev/null; then \
		: ; \
	elif $(PYTHON) -m venv --without-pip $(VENV) 2>/dev/null; then \
		: ; \
	else \
		echo "Falling back to virtualenv (installed via system pip $(PYTHON))"; \
		$(PYTHON) -m pip install --user virtualenv >/dev/null; \
		$(PYTHON) -m virtualenv $(VENV); \
	fi; \
	if [ ! -x "$(PIP)" ]; then \
		if [ -x "$(VENV)/bin/pip3" ]; then ln -s pip3 "$(PIP)"; fi; \
	fi; \
	if [ ! -x "$(PIP)" ]; then echo "pip not found in $(VENV)"; exit 1; fi; \
	"$(PIP)" install --upgrade pip >/dev/null

clean:
	rm -rf $(VENV)
