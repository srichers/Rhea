PYTHON ?= python3
VENV ?= .venv
REQS ?= requirements.txt
PIP ?= $(VENV)/bin/pip

.PHONY: venv clean train plot

venv: $(PIP)
	$(PIP) install -r $(REQS)

$(PIP):
	@echo "Creating virtual environment in $(VENV) using virtualenv..."
	@PIP_BREAK_SYSTEM_PACKAGES=1 $(PYTHON) -m pip install --user --upgrade virtualenv
	@$(PYTHON) -m virtualenv $(VENV)
	@if [ ! -x "$(PIP)" ] && [ -x "$(VENV)/bin/pip3" ]; then ln -s pip3 "$(PIP)"; fi
	@if [ ! -x "$(PIP)" ]; then echo "pip not found in $(VENV)"; exit 1; fi
	@$(PIP) install --upgrade pip

clean:
	rm -rf $(VENV)

train: venv
	$(VENV)/bin/python jmcguig_tests/run_many_models.py $(ARGS)

plot: venv
	@LOSS_PATH=$${LOSS:-loss.dat}; \
	OUT_PATH=$${OUT:-loss.png}; \
	if [ ! -f "$$LOSS_PATH" ]; then \
		echo "LOSS file not found: $$LOSS_PATH (set LOSS=/path/to/loss.dat)"; \
		exit 1; \
	fi; \
	echo "Plotting $$LOSS_PATH -> $$OUT_PATH"; \
	LOSS="$$LOSS_PATH" OUT="$$OUT_PATH" $(VENV)/bin/python - <<'PY'
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

loss_path = os.environ.get("LOSS", "loss.dat")
out_path = os.environ.get("OUT", "loss.png")

if not os.path.exists(loss_path):
    print(f"loss file not found: {loss_path}", file=sys.stderr)
    sys.exit(1)

with open(loss_path, "r", encoding="utf-8") as f:
    lines = [ln.strip() for ln in f if ln.strip()]
if len(lines) < 2:
    print(f"loss file is empty: {loss_path}", file=sys.stderr)
    sys.exit(1)

header = [part.split(":", 1)[-1] for part in lines[0].split("\t") if part]
rows = []
for line in lines[1:]:
    parts = [p for p in line.split("\t") if p]
    if len(parts) != len(header):
        continue
    rows.append([float(p) for p in parts])

if not rows:
    print(f"no data rows parsed from {loss_path}", file=sys.stderr)
    sys.exit(1)

data = np.asarray(rows, dtype=float)

def col(name):
    try:
        idx = header.index(name)
    except ValueError:
        return None
    return data[:, idx]

epoch = col("epoch")
train_loss = col("train_loss")
test_loss = col("test_loss")
learning_rate = col("learning_rate")

plt.figure(figsize=(8, 4.5))
if train_loss is not None:
    plt.plot(epoch, train_loss, label="train_loss")
if test_loss is not None:
    plt.plot(epoch, test_loss, label="test_loss")
plt.yscale("log")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid(True, which="both", alpha=0.3)
plt.legend(loc="best")

if learning_rate is not None:
    ax2 = plt.gca().twinx()
    ax2.plot(epoch, learning_rate, color="gray", linestyle="--", label="learning_rate")
    ax2.set_ylabel("learning rate")
    ax2.set_yscale("log")
    ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig(out_path, dpi=150)
print(f"Wrote {out_path}")
PY
