import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as tq
import argparse
import os
import sys
import copy
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vit_model import get_base_model, buildQuant
from models.layers import replace_layernorm_to_fpga

model = get_base_model()
quant_model_prepared = buildQuant(model, use_hw=False)

state_dict = torch.load("/home/mmic/SJS/01_SW/01_ViT/models/CUS_ViT_QAT_fakequant_4.pt", map_location="cpu")
quant_model_prepared.load_state_dict(state_dict, strict=False)

quant_model_prepared.cpu().eval()
quant_model_int8 = tq.quantize_fx.convert_fx(quant_model_prepared)

torch.save(quant_model_int8.state_dict(), "/home/mmic/SJS/01_SW/01_ViT/models/vit_qat_int8_custom_4.pt")
print(f"[Save] INT8 converted model saved")
