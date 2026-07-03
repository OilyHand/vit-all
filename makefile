PYTHON        := python3
MODEL_PATH    := /home/xilinx/projects/checkpoints/vit/vit_qat_int8_custom.pt
LOG_PATH_CPU  := /home/xilinx/projects/vit_all/log/infer_log_cpu.csv
LOG_PATH_FPGA := /home/xilinx/projects/vit_all/log/infer_log_fpga.csv
INFER_PATH    := /home/xilinx/projects/vit_all/scripts/run_infer_int8.py
HW_PATH       := /home/xilinx/projects/vit_all/hardware/TB_TPU_BD_wrapper.xsa

BATCH ?= 8

infer:
	$(PYTHON) $(INFER_PATH) \
    --model_path "$(MODEL_PATH)" \
    --log_path   "$(LOG_PATH_CPU)" \
    --batch_size 8