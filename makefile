PYTHON        := python
MODEL_PATH    := models/vit_qat_int8_custom.pt
INFER_PATH    := scripts/infer.py
HW_PATH       := ../hardware/FINAL.xsa

BATCH ?= 8

infer:
	sudo -E $(PYTHON) $(INFER_PATH) \
    --model_path "$(MODEL_PATH)" \
	--hw_path    "$(HW_PATH)" \
    --batch_size $(BATCH)
