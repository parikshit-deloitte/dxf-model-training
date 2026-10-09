PY := PYTHONPATH=src .venv/bin/python
BACKEND ?= auto
HOST ?= 127.0.0.1
DS ?= $(shell ls -d data/datasets/ds-* | tail -1)
RUN ?=
ZRUN ?= runs/zeroshot-$(notdir $(DS))

test:        ; $(PY) -m pytest -q tests
stage-a:     ; $(PY) -m dxftrain.data.build && $(PY) -m dxftrain.data.duplicate_report && $(PY) -m dxftrain.data.make_dataset && $(PY) -m dxftrain.data.labelling_sheet
stage-b:     ; $(PY) -m dxftrain.baselines.stage_b --dataset $(DS)
smoke:       ; $(PY) -m dxftrain.train.mlx_train --dataset $(DS) --seed 20261007 --smoke
smoke-cuda:  ; $(PY) -m dxftrain.train.train_lora --dataset $(DS) --seed 20261007 --smoke
stage-c:     ; $(PY) -m dxftrain.train.mlx_train --dataset $(DS) --seed 20261007 && $(PY) -m dxftrain.train.mlx_train --dataset $(DS) --seed 7
stage-d:     ; $(PY) -m dxftrain.calibrate.stage_d --dataset $(DS) --run $(RUN)
zero-shot-d: ; $(PY) -m dxftrain.calibrate.stage_d --dataset $(DS) --zero-shot
package:     ; $(PY) -m dxftrain.package.package --run $(RUN)
stage-e:     ; $(PY) -m dxftrain.gate.gate --dataset $(DS) --run $(RUN) --zero-shot-run $(ZRUN) --served models/$(notdir $(RUN))/served
serve:       ; .venv/bin/uvicorn serve.wrapper.app:app --host $(HOST) --port 8090
pipeline:    ; BACKEND=$(BACKEND) nohup scripts/run_pipeline.sh $(DS) >> reports/pipeline/runner.out 2>&1 &
backend:     ; @$(PY) -m dxftrain.infer.backend
retrain-check: ; $(PY) -m dxftrain.feedback.feedback retrain-check
.PHONY: test stage-a stage-b smoke smoke-cuda stage-c stage-d zero-shot-d package stage-e serve retrain-check pipeline backend
