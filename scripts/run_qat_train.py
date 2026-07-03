import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.ao.quantization as tq
from tqdm import tqdm
import argparse
import os
import sys
import copy
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from data import get_dataloaders
from models.vit_model import get_base_model, buildQuant
from models.quantization import apply_mixed_sparsity_
from models.layers import replace_layernorm_to_fpga

def parse_args():
    parser = argparse.ArgumentParser(description="ViT QAT Training with KD & Sparsity")
    parser.add_argument("--pretrained_path", type=str, default="models/vit_b16_cifar100_fp_extra.pth", help="Path to pretrained FP32 weights")
    parser.add_argument("--save_path_qat", type=str, default="models/CUS_ViT_QAT_fakequant.pt", help="Path to save QAT checkpoint")
    parser.add_argument("--save_path_int8", type=str, default="models/CUS_ViT_QAT_int8_converted.pt", help="Path to save converted INT8 model")
    parser.add_argument("--log_path", type=str, default="log/qat_train.csv", help="Path to training log")
    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Sparsity Hyperparameters
    parser.add_argument("--dense_ratio", type=float, default=0.5)
    parser.add_argument("--dense_sparsity", type=float, default=0.0)
    parser.add_argument("--sparse_sparsity", type=float, default=0.6)
    
    return parser.parse_args()

def apply_mlp_sparsity(model, dense_ratio, dense_sparsity, sparse_sparsity):
    """
    ViT Encoder Block 내의 MLP 레이어(Linear)들에 Mixed Sparsity를 직접 적용합니다.
    (경로 탐색 오류 방지를 위해 apply_mixed_sparsity_ 헬퍼 대신 직접 구현)
    """
    print(f"[Info] Applying MLP Sparsity (Ratio: {dense_ratio}, Dense: {dense_sparsity}, Sparse: {sparse_sparsity})...")
    count = 0
    
    # 모델의 모든 모듈을 순회하며 MLP Linear 레이어를 찾습니다.
    for name, module in model.named_modules():
        # MLP 내부의 Linear 레이어인지 확인 (보통 이름이 0, 3 등으로 끝남)
        if "mlp" in name and isinstance(module, (nn.Linear, torch.ao.nn.qat.Linear)):
            # 확실한 타겟팅을 위해 이름 끝자리 확인 (ViT 구조 의존)
            if name.endswith("0") or name.endswith("3") or "linear" in name:
                with torch.no_grad():
                    W = module.weight.data
                    out_ch, in_ch = W.shape
                    
                    # 마스크 생성 로직
                    split = int(in_ch * dense_ratio)
                    mask = torch.ones_like(W)
                    
                    if split > 0:
                        mask[:, :split] = (torch.rand(out_ch, split, device=W.device) > dense_sparsity).float()
                    if split < in_ch:
                        mask[:, split:] = (torch.rand(out_ch, in_ch - split, device=W.device) > sparse_sparsity).float()
                    
                    # 가중치에 마스크 적용 (In-place)
                    module.weight.mul_(mask)
                    
                    # (선택) 나중에 재확인을 위해 버퍼로 등록
                    module.register_buffer("_sparsity_mask", mask)
                    
                count += 1
                
    print(f"[Info] Applied sparsity to {count} MLP Linear layers.")

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # 1. Data Loader
    train_loader, test_loader, _ = get_dataloaders(batch_size=args.batch_size)
    print(f"[Data] Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # 2. Teacher Model (FP32)
    print(f"[Init] Loading Teacher Model from {args.pretrained_path}")
    teacher = get_base_model(num_classes=100)
    if os.path.exists(args.pretrained_path):
        state_dict = torch.load(args.pretrained_path, map_location="cpu")
        teacher.load_state_dict(state_dict)
    else:
        print(f"[Warning] Pretrained file not found! Using random weights for Teacher.")
    
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # 3. Student Model (QAT Prepared)
    # Teacher의 가중치를 복사하여 시작
    student_base = copy.deepcopy(teacher).cpu()
    
    # buildQuant 내부에서 replace(MHA -> QuantMHA) 및 prepare_qat_fx 수행
    # 주의: use_hw=False로 설정하여 학습 중에는 FakeQuant만 수행 (나중에 변환)
    print("[Init] Building QAT Model...")
    quant_model = buildQuant(student_base, use_hw=False)
    
    # 4. Apply MLP Sparsity
    apply_mlp_sparsity(quant_model, args.dense_ratio, args.dense_sparsity, args.sparse_sparsity)
    
    quant_model.to(device)

    state_dict = torch.load("/home/mmic/SJS/01_SW/01_ViT/models/vit_qat_fakequant.pt", map_location=device)
    quant_model.load_state_dict(state_dict, strict=False)

    n_ln = replace_layernorm_to_fpga(quant_model, False)
    print(f"[Info] Replaced {n_ln} LayerNorm layers.")
    
    # 5. Optimizer & Loss
    optimizer = optim.AdamW(quant_model.parameters(), lr=args.lr, weight_decay=0.0)
    # KD Hyperparams
    T = 2.0
    alpha = 0.5

    # 6. Training Loop (QAT + KD)
    print("===============================================================")
    print(f" ***         Starting QAT Training for {args.epochs:3d} epochs         ***")
    print("===============================================================")
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        quant_model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, (imgs, labels) in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Teacher Forward
            with torch.no_grad():
                t_logits = teacher(imgs)
            
            # Student Forward (Fake Quantized)
            s_logits = quant_model(imgs)
            
            # KD Loss
            kd_loss = F.kl_div(
                F.log_softmax(s_logits / T, dim=1),
                F.softmax(t_logits / T, dim=1),
                reduction="batchmean"
            ) * (T * T)
            
            # CE Loss
            ce_loss = F.cross_entropy(s_logits, labels)
            
            # Total Loss
            loss = alpha * kd_loss + (1.0 - alpha) * ce_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = s_logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            # if (batch_idx + 1) % 50 == 0:
            #     print(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
            #           f"Loss: {loss.item():.4f} (KD: {kd_loss.item():.4f}, CE: {ce_loss.item():.4f})")
            current_loss = total_loss / (batch_idx + 1)
            current_acc = correct / total
            
            # [수정] 진행 바 오른쪽에 실시간 수치 표시 (Loss, Acc, KD, CE)
            pbar.set_postfix({
                "Loss": f"{current_loss:.4f}",
                "Acc": f"{current_acc:.4f}",
                "KD": f"{kd_loss.item():.4f}",
                "CE": f"{ce_loss.item():.4f}"
            })

        # print("\n[Eval] Evaluating Fake-Quantized Model on GPU...")
        # quant_model.eval()
        # correct = 0
        # total = 0
        # with torch.no_grad():
        #     for imgs, labels in test_loader:
        #         imgs = imgs.to(device)
        #         labels = labels.to(device)
        #         preds = quant_model(imgs)
        #         correct += (preds.argmax(dim=1) == labels).sum().item()
        #         total += labels.size(0)
        #         print(f"Accuracy: {correct / total:.4f}")
        
        # acc = correct / total
        # print(f"Epoch {epoch+1} Accuracy: {acc:.4f} (Best: {best_acc:.4f})")
        # if acc > best_acc:
            # best_acc = acc
            # 파일명 뒤에 _best를 붙여서 저장

        avg_loss = total_loss / len(train_loader)
        train_acc = correct / total
        print(f"Epoch [{epoch+1}] Done. Avg Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}")

        best_path = args.save_path_qat.replace(".pt", f"_{epoch+1}.pt")
        torch.save(quant_model.state_dict(), best_path)
        print(f"[Save] New Best Model saved to {best_path} (Acc: {train_acc:.4f})")

        # with open(args.log_path, "a") as f:
        #     ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #     f.write(f"{ts},{epoch+1},{args.dense_ratio},{args.dense_sparsity},{args.sparse_sparsity},{acc:.4f}\n")



    # 7. Evaluation (Fake Quantized Model)
    print("\n[Eval] Evaluating Fake-Quantized Model on GPU...")
    quant_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            preds = quant_model(imgs)
            correct += (preds.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            
            print(f"Accuracy: {correct / total:.4f}")

    acc = correct / total
    print(f"Accuracy (QAT Fake-Quant): {acc:.4f}")

    # 8. Convert to INT8 & Save
    # 변환은 CPU에서 수행해야 함
    print("\n[Convert] Converting to INT8 Model (CPU)...")
    quant_model.cpu().eval()
    quant_model_int8 = tq.quantize_fx.convert_fx(quant_model)
    
    # 모델 저장
    os.makedirs(os.path.dirname(args.save_path_qat), exist_ok=True)
    
    # 1) QAT 학습된 가중치 (Fake Quant 상태, 재학습 가능)
    torch.save(quant_model.state_dict(), args.save_path_qat)
    print(f"[Save] QAT checkpoint saved to {args.save_path_qat}")
    
    # 2) INT8 변환된 모델 (실제 배포용)
    torch.save(quant_model_int8.state_dict(), args.save_path_int8)
    print(f"[Save] INT8 converted model saved to {args.save_path_int8}")

    # 로그 저장
    with open(args.log_path, "a") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{ts},{args.dense_ratio},{args.dense_sparsity},{args.sparse_sparsity},{acc:.4f}\n")
    print(f"[Log] Result appended to {args.log_path}")

if __name__ == "__main__":
    main()
