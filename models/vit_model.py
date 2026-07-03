import torch
import torch.nn as nn
import torchvision.models as tvm
import torch.ao.quantization as tq
import os

from config import config
from models.quantization import buildQuant
from models.layers import QuantLayerNormFPGA, FPGAManager, replace_layernorm_to_fpga

# =========================================================================
# [1] Base Model Factory
# =========================================================================

def get_base_model(num_classes=100, weights=None):
    model = tvm.vit_b_16(weights=weights, num_classes=100)

    return model


# =========================================================================
# [2] QAT Model Factory (For Training)
# =========================================================================

def get_qat_model_for_training(num_classes=100, use_hw=False, backend="fbgemm"):

    base_model = get_base_model(num_classes=num_classes, weights=None)
    qat_model = buildQuant(base_model, use_hw=use_hw, qbackend=backend)
    
    return qat_model


# =========================================================================
# [3] Quantized Model Factory (For Inference)
# =========================================================================

def get_quant_model(checkpoint_path, num_classes=100, qbackend="qnnpack", batch_size= 1, hw=None, use_hw=False, device="cpu"):
    base_model = get_base_model(num_classes=num_classes)
    quant_model_prepared = buildQuant(base_model, batch_size=batch_size, use_hw=False, qbackend=qbackend)
    quant_model_int8 = tq.quantize_fx.convert_fx(quant_model_prepared)

    print("dbg point 4")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    quant_model_int8.load_state_dict(state_dict, strict=False)
    
    print("dbg point 5")

    quant_model_int8.to(device).eval()
    
    return quant_model_int8
