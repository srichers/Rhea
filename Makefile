PYTHON ?= python3
VENV ?= .venv
REQS ?= requirements.txt
GET_PIP_URL ?= https://bootstrap.pypa.io/get-pip.py
GET_PIP ?= /tmp/get-pip.py

.PHONY: venv clean

venv: $(VENV)/bin/pip
	. $(VENV)/bin/activate && pip install -r $(REQS)

$(VENV)/bin/python:
	$(PYTHON) -m venv --without-pip $(VENV)

$(VENV)/bin/pip: $(VENV)/bin/python
	@echo "Bootstrapping pip into $(VENV)..."
	@($(VENV)/bin/python -m ensurepip --upgrade) || \
	  ( \
	    if command -v curl >/dev/null 2>&1; then curl -sSf $(GET_PIP_URL) -o $(GET_PIP); \
	    elif command -v wget >/dev/null 2>&1; then wget -q -O $(GET_PIP) $(GET_PIP_URL); \
	    else \
	      echo "Downloading get-pip.py with Python stdlib"; \
	      $(PYTHON) - <<'PY' ;\
import urllib.request, os, sys; url="$(GET_PIP_URL)"; path="$(GET_PIP)"; \
urllib.request.urlretrieve(url, path); \
print(f"Downloaded {path} from {url}") \
PY \
	    ; fi; \
	    $(VENV)/bin/python $(GET_PIP); \
	    rm -f $(GET_PIP); \
	  )
	@$(VENV)/bin/pip install --upgrade pip

clean:
	rm -rf $(VENV)
