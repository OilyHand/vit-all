PYTHON        := python
MODEL_PATH    := models/vit_qat_int8_custom.pt
INFER_PATH    := scripts/infer.py
HW_PATH       := hardware/TB_TPU_BD_wrapper.xsa

BATCH ?= 8

infer:
	$(PYTHON) $(INFER_PATH) \
    --model_path "$(MODEL_PATH)" \
    --batch_size 8
