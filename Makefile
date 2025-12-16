PYTHON ?= python3
VENV ?= .venv
REQS ?= requirements.txt
GET_PIP_URL ?= https://bootstrap.pypa.io/get-pip.py

.PHONY: venv clean

venv: $(VENV)/bin/pip
	. $(VENV)/bin/activate && pip install -r $(REQS)

$(VENV)/bin/python:
	$(PYTHON) -m venv --without-pip $(VENV)

$(VENV)/bin/pip: $(VENV)/bin/python
	@echo "Bootstrapping pip into $(VENV)..."
	@($(VENV)/bin/python -m ensurepip --upgrade || \
	  (curl -sSL $(GET_PIP_URL) -o /tmp/get-pip.py && $(VENV)/bin/python /tmp/get-pip.py && rm /tmp/get-pip.py))
	@$(VENV)/bin/pip install --upgrade pip

clean:
	rm -rf $(VENV)
