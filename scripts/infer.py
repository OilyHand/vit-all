import torch, torchvision
from   torch.utils.data import DataLoader, Subset
import glob
import argparse
import time
import os
import sys
import traceback
from datetime import datetime
from tqdm import tqdm
from pynq import Overlay, MMIO, allocate
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from models import get_quant_model
from models import *
from models.layers import replace_ln_to_fpga

import time
import pdb
from collections import defaultdict
from datetime import datetime

# === Hook 셋업 ===
block_times = defaultdict(list)
_block_start = {}

torch.backends.quantized.engine = "qnnpack"

def make_pre_hook(name):
    def hook(module, input):
        _block_start[name] = time.perf_counter()
    return hook

def make_post_hook(name):
    def hook(module, input, output):
        elapsed = time.perf_counter() - _block_start[name]
        block_times[name].append(elapsed)
    return hook

def parse_args():
    parser = argparse.ArgumentParser(description="ViT INT8 Inference Script")
    parser.add_argument("--model_path", type=str, default="models/vit_qat_int8_custom.pt", help="Path to the converted INT8 checkpoint")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,               help="Batch size for inference")
    parser.add_argument("--device",     type=str, default="cpu",                           help="Device to run inference on (cpu/cuda)")
    parser.add_argument("--log_path",   type=str, default="./log/infer_int8.csv",          help="Path to save inference logs")
    parser.add_argument("--use_hw",     action="store_true",                               help="Enable FPGA Hardware Acceleration")
    parser.add_argument("--hw_path", type=str, default="./hardware/TB_TPU_BD_wrapper.xsa", help="Path to hardware xsa file")
    return parser.parse_args()

BASE            = 0xA000_0000
MMIO_RANGE      = 0x1000
CSRA_CONTROL    = 0x00
SA_SOURCE1      = 0x04
SA_SOURCE2      = 0x08
SA_CONT1        = 0x0C
SA_CONT2        = 0x10
SA_DESTINATION  = 0x14
IRQ_CLEAR_OFF   = None
IRQ_CLEAR_VALUE = 0x1 

if __name__ == "__main__":
    args = parse_args()

    # -------------------------------------------------------------------------
    # 1. Setup Environment
    # -------------------------------------------------------------------------
    device = torch.device(args.device)
    print(f"[Init] Device: {device}")
    print(f"[Init] Loading Model from: {args.model_path}")
    
    # -------------------------------------------------------------------------
    # 2. Load Data
    # -------------------------------------------------------------------------
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), 
                                         (0.2675, 0.2565, 0.2761))
    ])

    test_set = torchvision.datasets.CIFAR100(
        root="./data",
        train=False,
        download=False,
        transform=preprocess)

    indices = list(range(4096))
    test_set = Subset(test_set, indices)

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False)

    # -------------------------------------------------------------------------
    # 3. Load Model
    # -------------------------------------------------------------------------
    hw = FPGAManager(args.hw_path)
    # hw = None
    model = get_quant_model(checkpoint_path=args.model_path,
                            num_classes=100,
                            qbackend="qnnpack",
                            batch_size=args.batch_size,
                            hw=hw,
                            use_hw=args.use_hw,
                            device=args.device)
    
    # traced = torch.fx.symbolic_trace(model)
    # breakpoint()
    
    # for node in model.graph.nodes:
    #     print(node.name, node.target)
    #     breakpoint()

    try:
        # model = transform_mha_to_tpu(model, hw)
        # model = transform_conv_to_tpu(quantized_model,hw)
        # model = transform_quantized_model_to_tpu(quantized_model,hw)
        model = replace_ln_to_fpga(model, hw)
        
        import gc
        gc.collect()

    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
    
    except KeyboardInterrupt:
        del hw
        print("[Keyboard Interrupt] Exit")
        exit()

    # -------------------------------------------------------------------------
    # 4. Inference Loop
    # -------------------------------------------------------------------------
    import re

    blocks_to_hook = []
    for name, module in model.named_modules():
        if re.match(r'^encoder\.layers\.encoder_layer_\d+$', name):
            blocks_to_hook.append((name, module))

    print(f"Hooked {len(blocks_to_hook)} transformer blocks")

    hooks = []
    for name, block in blocks_to_hook:
        h1 = block.register_forward_pre_hook(make_pre_hook(name))
        h2 = block.register_forward_hook(make_post_hook(name))
        hooks.extend([h1, h2])

    print(f"Registered {len(hooks)} hooks")

    correct = 0
    total = 0
    total_inference_time = 0.0

    
    print("===============================================================")
    print(" ***             Starting Inference on TestSet             *** ")
    print("===============================================================")


    # from pynq.pmbus import DataRecorder
    # from pynq import get_rails

    # rails = get_rails()
    # recorder = DataRecorder( rails['12V'].power, rails['INT'].power, rails['1V2'].power, rails['1V8'].power) 

    # with recorder.record(0.001):
    #     time.sleep(2)  # 아무것도 안하는 상태

    #     idle_df = recorder.frame
    #     idle_12v = idle_df['12V_power'].mean()
    #     idle_int = idle_df['INT_power'].mean()
    #     idle_1v2 = idle_df['1V2_power'].mean()
    #     idle_1v8 = idle_df['1V8_power'].mean()

    try:
        with torch.inference_mode():
            warmup_done = False
            total_batches = len(test_loader)
            batch_count = 0
            for imgs, labels in test_loader:
                batch_count += 1
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                # 시간 측정
                start_time = time.perf_counter()
                preds = model(imgs)
                end_time = time.perf_counter()
            
                # df = recorder.frame
                # power_12v  = df['12V_power'].mean()
                # power_int  = df['INT_power'].mean()
                # power_1v2  = df['1V2_power'].mean()
                # power_1v8  = df['1V8_power'].mean()
                

                # 배치 처리 시간 누적
                batch_time = end_time - start_time
                total_inference_time += batch_time
                best_latency = float('inf')

                # 정확도 계산
                pred_cls = preds.argmax(dim=1)
                correct += (pred_cls == labels).sum().item()
                total += labels.size(0)
                
                # 현재 정확도와 평균 레이턴시 계산
                current_acc = correct / total
                latency_sec = (batch_time)
                avg_latency_sec = (total_inference_time / total)*1000             
                best_latency  = min(best_latency,  (batch_time * 1000) / labels.size(0)) 

                print(f"Acc: {current_acc:.4f} | "
                  f"AvgTime: {avg_latency_sec:.2f}ms | "
                  f"BestTime: {best_latency:.2f}ms | ")
                #   f"12V: {power_12v:.2f}W | "
                #   f"INT: {power_int:.2f}W | "
                #   f"1V2: {power_1v2:.2f}W | "
                #   f"1V8: {power_1v8:.2f}W | ") 

            # === 최종 block별 평균 출력 ===
            print("\n=== Per-block timing (avg) ===")
            for name, _ in blocks_to_hook:
                ts = block_times[name]
                if ts:
                    avg_ms = sum(ts) / len(ts) * 1000
                    print(f"  {name}: {avg_ms:.3f} ms")
            
            # 전체 transformer 평균
            all_block_times = [t for ts in block_times.values() for t in ts]
            if all_block_times:
                print(f"\nAll blocks avg: {sum(all_block_times)/len(all_block_times)*1000:.3f} ms")

    except KeyboardInterrupt:
        del hw
        print("[Keyboard Interrupt] Exit")
        exit()

    except Exception as e:
        traceback.print_exc()
    
    finally:
        for h in hooks:
            h.remove()
        print(f"\n[Cleanup] Removed {len(hooks)} hooks")

    # -------------------------------------------------------------------------
    # 5. Final Report
    # -------------------------------------------------------------------------
    final_acc = correct / total
    final_avg_latency_ms = (total_inference_time / total) * 1000
    
    print("\n===============================================================")
    print(f" [Result] Final Accuracy: {final_acc:.4f} ({correct}/{total})")
    print(f" [Result] Avg Latency   : {final_avg_latency_ms:.4f} ms/sample")
    print(f" [Result] Total Time    : {total_inference_time:.4f} sec")
    print("===============================================================")
