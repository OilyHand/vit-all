import torch
import os

def inspect_model_weights(file_path):
    print(f"--- Loading model from: {file_path} ---\n")
    
    try:
        loaded_data = torch.load(file_path, map_location='cpu', weights_only=False)
        
        # 로드된 데이터가 state_dict(딕셔너리)인지 모델 객체인지 확인
        if isinstance(loaded_data, dict):
            # 'model', 'state_dict' 등의 키로 감싸져 있는 경우 처리
            if 'state_dict' in loaded_data:
                state_dict = loaded_data['state_dict']
            elif 'model' in loaded_data:
                state_dict = loaded_data['model']
            else:
                state_dict = loaded_data
        elif hasattr(loaded_data, 'state_dict'):
            state_dict = loaded_data.state_dict()
        else:
            print("Error: 모델의 state_dict를 찾을 수 없습니다.")
            return

        # 각 레이어/파라미터 순회
        for name, tensor in state_dict.items():
            if not torch.is_tensor(tensor):
                continue

            print(f"1) Name: {name}")
            print(f"2) Dtype: {tensor.dtype}")

            if tensor.is_quantized:
                if tensor.qscheme() in (torch.per_channel_affine, torch.per_channel_symmetric):
                    print(f"3) Scale: (Per-Channel) {tensor.q_per_channel_scales()}")
                    print(f"4) Zero Point: (Per-Channel) {tensor.q_per_channel_zero_points()}")
                else:
                    print(f"3) Scale: {tensor.q_scale()}")
                    print(f"4) Zero Point: {tensor.q_zero_point()}")
                
                print(f"5) Min: {tensor.dequantize().min().item():.6f} (dequantized)")
                print(f"   Max: {tensor.dequantize().max().item():.6f} (dequantized)")
                
            else:
                print("3) Scale: N/A (Not Quantized)")
                print("4) Zero Point: N/A (Not Quantized)")
                
                # 일반 텐서의 min/max
                # 데이터가 비어있거나 복소수일 경우 예외 처리 가능
                if tensor.numel() > 0 and not tensor.is_complex():
                    print(f"5) Min: {tensor.min().item():.6f}")
                    print(f"   Max: {tensor.max().item():.6f}")
                else:
                    print("5) Min/Max: N/A")

            print("-" * 50)

    except Exception as e:
        print(f"오류 발생: {e}")

# 실행 예시
if __name__ == "__main__":
    # 여기에 분석할 .pt 또는 .pth 파일 경로를 입력하세요
    MODEL_PATH = "/home/mmic/SJS/01_SW/01_ViT/models/vit_qat_int8_custom_mod.pt" 
    
    if os.path.exists(MODEL_PATH):
        inspect_model_weights(MODEL_PATH)
    else:
        print(f"파일을 찾을 수 없습니다: {MODEL_PATH}")