PYTHON := python3
VENV := .venv
REQS := requirements.txt

.PHONY: venv clean

venv: $(VENV)/bin/activate

$(VENV)/bin/activate: $(REQS)
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install --upgrade pip
	. $(VENV)/bin/activate && pip install -r $(REQS)
	@touch $(VENV)/bin/activate

clean:
	rm -rf $(VENV)
