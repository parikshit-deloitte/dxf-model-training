# dxf-model-training

Standalone training repo for the **per-object classifier**: given name-free evidence about one object (closed
polygon, dimension, text) from a DXF with no usable layer names, return `{"category": <Manual category | unknown>}`.
It never imports the scrutiny system and holds no credentials; its only link to it is the served URL.

Design: `../Model-Train/TRAINING_DESIGN.md`. Stage reports: `reports/`.

| Stage | Command | Output |
|---|---|---|
| A data | `make stage-a` | `data/datasets/ds-*/` (read-only), `DATACARD.md`, `data/plot_match_report.json`, `data/sheets/sheet-v*/` |
| B baselines | `make stage-b` | `reports/stage_b/<ds>/` |
| C fine-tune | `make smoke`, then `make stage-c` (MLX) or `train_lora.py` (CUDA) | `runs/<id>/` |
| D confidence | `make stage-d RUN=runs/<id>`, `make zero-shot-d` | `runs/<id>/calibration.json` |
| E gate | `make package RUN=…`, `make stage-e RUN=…` | `models/<id>/gate_report.json`, `MODEL_CARD.md` |
| F serve | `python -m dxftrain.registry.registry register/promote/rollback`, `make serve` | `models/registry.json`, POST `/classify`, POST `/classify_drawing` |
| G feedback | `python -m dxftrain.feedback.feedback ingest-corrections|ingest-sheet|retrain-check` | `data/feedback/*-v*/` |

## Run it on any machine (Apple Silicon or NVIDIA)

The same pipeline runs on two backends, chosen automatically (`make backend` prints it):

| Backend | Where | Trainer | Scorer (Stages B3, D, E, server) |
|---|---|---|---|
| `mlx` | Mac with Apple Silicon | `train/mlx_train.py` (MLX 4-bit) | `infer/scorer.py` |
| `cuda` | NVIDIA GPU, e.g. AWS g4dn.xlarge (T4 16 GB) | `train/train_lora.py` (QLoRA, fp16 on a T4) | `infer/torch_scorer.py` |

A run, its calibration and its registry entry record the backend; a model is always scored by the backend it was
trained with. Setup on a GPU server:

```bash
git clone <this repo> && cd dxf-model-training
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch && pip install -e ".[cuda,serve,dev]"
make test                                   # no GPU needed
# copy the dataset (never via git): data/datasets/ds-…, configs/test_projects.v1.json, reports/stage_b/ds-…
make smoke-cuda DS=data/datasets/ds-…       # 1 step; measure s/step here before the full run
make pipeline BACKEND=cuda DS=data/datasets/ds-…   # background; tail -f reports/pipeline/<ds>/cuda/progress.log
```

Serve it and let the scrutiny system call it: `CLASSIFIER_API_KEY=<long random> MODEL_ID=<run id> make serve`
(127.0.0.1:8090; `HOST=0.0.0.0` only behind HTTPS). In the scrutiny system set `CLASSIFIER_URL` and
`CLASSIFIER_API_KEY`. Reach it through an SSH tunnel, a private network, or HTTPS with the key; never a bare port.

**Everything B→E in one go:** `make pipeline DS=data/datasets/ds-…` (or `BACKEND=… SEEDS="20261007 7"
scripts/run_pipeline.sh …`) — one step at a time (one GPU), resumable, logs and `progress.log` in
`reports/pipeline/<ds>/<backend>/`. The first seed is trained, calibrated, packaged and registered first (usable in
trial mode), then gated; the registry promotes it only on a PASS gate against the current active model.

**Plot rule.** A file supplies objects when exactly one polygon is within 2% of `PLOT_AREA_M2`, or (rule H1,
approved 2026-10-09, `configs/data.yaml duplicate_policy: H1`) when all matching polygons are the same outline drawn
more than once. Extractor 1.1.0.

**Serving for the scrutiny system.** `POST /classify_drawing` takes a DXF file as the request body and returns, per
closed polygon, `{handle, category, confidence}` plus the model id, prompt hash, threshold and the model's gate
verdict. It runs the training extractor and feature steps, so served features are the trained ones. A model that is
not PASS can be served only by naming it: `MODEL_ID=<id> make serve` (trial mode; the scrutiny system then marks
every row that depends on it "To be verified by official").

Never: put a layer name in a prompt; split by file instead of project; change `prompts/classify_system.v1.txt`
without a new version (every dataset/run/model pins its hash); mix synthetic rows into real counts; promote
without a PASS gate against the currently active model. Real drawings are local data and never committed.
