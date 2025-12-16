PYTHON ?= python3
VENV ?= .venv
REQS ?= requirements.txt
GET_PIP_URL ?= https://bootstrap.pypa.io/get-pip.py
GET_PIP ?= $(VENV)/get-pip.py
PIP ?= $(VENV)/bin/pip
PIP3 ?= $(VENV)/bin/pip3

.PHONY: venv clean

venv: $(PIP)
	$(PIP) install -r $(REQS)

$(VENV)/bin/python:
	$(PYTHON) -m venv --without-pip $(VENV)

$(PIP): $(VENV)/bin/python
	@echo "Bootstrapping pip into $(VENV)..."
	@($(VENV)/bin/python -m ensurepip --upgrade) || ( \
		set -e; \
		if command -v curl >/dev/null 2>&1; then \
			curl -sSf $(GET_PIP_URL) -o $(GET_PIP); \
		elif command -v wget >/dev/null 2>&1; then \
			wget -q -O $(GET_PIP) $(GET_PIP_URL); \
		else \
			echo "Downloading get-pip.py with Python stdlib"; \
			$(PYTHON) -c "import urllib.request; url='$(GET_PIP_URL)'; path='$(GET_PIP)'; urllib.request.urlretrieve(url, path); print('Downloaded %s from %s' % (path, url))"; \
		fi; \
		if [ ! -f $(GET_PIP) ]; then echo 'Failed to download get-pip.py'; exit 1; fi; \
		$(VENV)/bin/python $(GET_PIP); \
	)
	@if [ ! -x $(PIP) ] && [ -x $(PIP3) ]; then ln -s pip3 $(PIP); fi
	@$(PIP) install --upgrade pip

clean:
	rm -rf $(VENV)
