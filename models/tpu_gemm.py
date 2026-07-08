from pynq import Overlay, MMIO, allocate, Interrupt
from models.def_utils import *
import numpy as np
import time
import glob
import os, struct, select, threading, torch ,asyncio
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.ao.nn.quantized.modules.linear as qlinear
import ctypes
import os
# -------------------------------
# MMIO / Register map (캡처 기준)
# -------------------------------
BASE = 0xA000_0000
MMIO_RANGE = 0x1000

CSRA_CONTROL   = 0x00
SA_SOURCE1     = 0x04
SA_SOURCE2     = 0x08
SA_CONT1       = 0x0C
SA_CONT2       = 0x10
SA_DESTINATION = 0x14

IRQ_CLEAR_OFF   = None   # 예: 0x38
IRQ_CLEAR_VALUE = 0x1    # 예: W1C면 1

import struct

def float_to_hex(f):
    # float → bytes → hex
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])


def preprocess_weight_for_tpu(weight_tensor, TRANS = True):
    # 🎯 수정 포인트: 입력이 텐서인지 넘파이인지 체크해서 안전하게 w_np 확보
    if hasattr(weight_tensor, 'detach'):
        w_np = weight_tensor.T.detach().cpu().numpy() if TRANS else weight_tensor.detach().cpu().numpy()
    else:
        w_np = weight_tensor.T if TRANS else weight_tensor
    # 이제 W는 항상 Numpy 배열이므로 .T (Transpose)가 안전하게 작동합니다.
    W = w_np
    remainder = w_np.shape[1] % 16
    if remainder != 0:
        padding_size = 16 - remainder
        # 가로 방향(axis=1)으로 0을 붙여줌
        W = np.pad(w_np, ((0, 0), (0, padding_size)), mode='constant', constant_values=0)

    #print(f" - Original Shape (In, Out): ({w_np.shape})")
    #print(f" - Padding Added: {W.shape} channels")

    #blocks = []
    #for i in range(0, W.shape[1], 16):
    #    blocks.append(W[:, i:i+16])
    #W_concat = np.concatenate(blocks, axis=0)
    rows, cols = w_np.shape
    W_concat = w_np.reshape(rows, cols // 16, 16) .transpose(1, 0, 2)  .reshape(-1, 16)
    # 또는
    W_concat = w_np.T.reshape(cols // 16, 16, rows) .transpose(0, 2, 1) .reshape(-1, 16)

    return w_np, W_concat

def find_scale_in_args(args, model):
    """args에서 scale get_attr 노드 찾기"""
    for arg in args:
        if hasattr(arg, 'op') and arg.op == 'get_attr':
            try:
                val = model.get_buffer(arg.target)
                # scalar tensor이면 scale!
                if val.numel() == 1:
                    return float(val)
            except:
                continue
    return None

def find_output_scale(node, model, debug=True):
    # matmul 노드는 args[2]=scale, args[3]=zero_point를 직접 들고 있음
    try:
        scale_node = node.args[2]
        zp_node    = node.args[3]

        if isinstance(scale_node, torch.fx.Node) and scale_node.op == 'get_attr':
            out_scale = float(getattr(model, scale_node.target).item())
        elif isinstance(scale_node, (float, int)):
            out_scale = float(scale_node)
        else:
            raise ValueError(f'scale 타입 미지원: {type(scale_node)}')

        if isinstance(zp_node, torch.fx.Node) and zp_node.op == 'get_attr':
            out_zp = int(getattr(model, zp_node.target).item())
        elif isinstance(zp_node, (float, int)):
            out_zp = int(zp_node)
        else:
            raise ValueError(f'zp 타입 미지원: {type(zp_node)}')

        if debug:
            print(f'✅ {node.name}: out_scale={out_scale}, out_zp={out_zp}')
        return out_scale, out_zp

    except Exception as e:
        if debug:
            print(f'❌ {node.name}: output scale 못 찾음 → {e}')
            print(f'  node.args: {node.args}')
            breakpoint()
        return None, None

def find_input_scale(node, model):
    curr = node.args[0]

    skip_ops = ['reshape', 'view', 'transpose',
                'permute', 'contiguous', 'flatten']
    while curr is not None:
        if any(op in str(curr.target) for op in skip_ops):
            curr = curr.args[0]
            continue

        if curr.op == 'call_function':
            fname = str(curr.target)

            # matmul 발견!
            if 'matmul' in fname or 'quantize_per_tensor' in fname:
                scale = find_scale_in_args(curr.args, model)
                if scale is not None:
                    return scale

        if curr.op == 'call_module':
            submod = model.get_submodule(curr.target)
            if hasattr(submod, 'scale'):
                return float(submod.scale)

        if curr.op == 'get_attr':
            return float(model.get_buffer(curr.target))

        curr = curr.args[0] if curr.args else None

    return None

def transform_quantized_model_to_tpu(model, hw):
    graph = model.graph
    # 1. 대상 노드(matmul) 수집
    nodes_to_replace1 = []
    for k in graph.nodes:
        if k.op == 'call_module':
            try:
                submod = model.get_submodule(k.target)
                if isinstance(submod, qlinear.Linear):
                    nodes_to_replace1.append(k)
            except AttributeError:
                continue
    print(f"🔍 총 {len(nodes_to_replace1)}개의 matmul(call_module) 노드를 발견했습니다.")
    for node in nodes_to_replace1:
        # 2. 가중치 모듈 이름 추적
        submod = model.get_submodule(node.target)
        weight, bias = submod._packed_params._weight_bias()

        # [2] Output Scale/ZP (모듈 속성에 직접 있음)
        out_scale = submod.scale
        out_zp = submod.zero_point

        # [3] Input Scale (위에서 만든 안전한 함수 사용)
        # node.args[0]은 qkv의 입력인 ln_1이나 dropout일 것임
        input_scale = find_input_scale(node, model)
        print(input_scale)
        #input_node = node.args[0]
        #input_submod = model.get_submodule(input_node.target)
        x_scale = input_scale
        # 확인

        # 4. TPULinear 모듈 생성 및 가중치 로드
        tpu_linear = TPULinear(node.target,x_scale,weight, bias,out_scale,out_zp, hw)

        # 5. 모델에 등록 (이름은 중복 안 되게 유니크하게)
        tpu_module_name = f"tpu_{node.name}"
        setattr(model, tpu_module_name, tpu_linear)

        # 6. 그래프 노드 교체 (설계도 수정)
        with graph.inserting_before(node):
            # 입력 노드를 명확히 지정 (튜플 형태 유지)
            input_node = node.args[0]
            new_node = graph.call_module(tpu_module_name, args=(input_node,))

            # 7. 기존 노드의 사용처를 새 노드로 교체
            node.replace_all_uses_with(new_node)
        # 기존 matmul 노드 삭제
        graph.erase_node(node)

    # 7. 변경된 그래프로 모델 재컴파일
    model.graph.lint()
    model.recompile()

    print("\n🚀 모든 matmul 노드가 TPU 가속 노드로 교체되었습니다!")
    return model


class TPUPatchEmbedding(nn.Module):
    def __init__(self, name, x_scale, weight_tensor, bias_tensor, out_scale, out_zp, hw):
        super().__init__()
        self.hw        = hw
        self.name      = name
        self.out_scale = float(out_scale)
        self.out_zp    = out_zp

        self.P     = 16
        self.H     = 224
        self.W_img = 224
        self.C     = 3
        self.N     = (self.H // self.P) * (self.W_img // self.P)  # 196
        self.N_pad = (self.N + 7) // 8 * 8                        # 200
        self.K     = self.C * self.P * self.P                     # 768
        self.M     = self.N * self.hw.batch_size

        self.INTERRUPT1 = hw.ip_ol.axi_intc_0
        self.weight_ori = weight_tensor
        self.weight = weight_tensor.int_repr().detach().cpu().numpy().reshape(768, -1)
        if not hasattr(hw, 'irq_loop'):
            new_loop = asyncio.new_event_loop()
            hw.irq_loop = new_loop
            t = threading.Thread(target=start_irq_loop, args=(new_loop,), daemon=True)
            t.start()
            print("🌐 [INFO] Shared IRQ loop started.")

        # ── Weight 추출 (TPULinear와 동일) ─────────────────────
        if hasattr(self.weight_ori, 'int_repr'):
            w_np = self.weight_ori.int_repr().detach().cpu().numpy()  # [768, 3, 16, 16]
            self.w_scale      = self.weight_ori.q_per_channel_scales()
            self.w_zero_point = self.weight_ori.q_per_channel_zero_points()
        elif hasattr(self.weight_ori, 'detach'):
            w_np = self.weight_ori.detach().cpu().numpy()
            self.w_scale      = 1.0
            self.w_zero_point = 0
        else:
            w_np = self.weight_ori
            self.w_scale      = 1.0
            self.w_zero_point = 0

        # Conv → GEMM reshape: [768, 3, 16, 16] → [768, 768]
        w_np = w_np.reshape(w_np.shape[0], -1)

        self.out_C       = w_np.shape[0]   # 768
        self.in_features = w_np.shape[1]   # 768

        # ── m_scale_per_channel 계산 (TPULinear와 동일) ────────
        if hasattr(self.w_scale, 'detach'):
            w_scale_np = self.w_scale.detach().cpu().numpy().astype(np.float32)
        else:
            w_scale_np = np.asarray(self.w_scale, dtype=np.float32)

        if w_scale_np.ndim == 0:
            w_scale_np = np.full(self.out_C, float(w_scale_np), dtype=np.float32)

        x_scale_f   = float(x_scale)
        out_scale_f = float(self.out_scale)
        m_scale_per_channel = (x_scale_f * w_scale_np / out_scale_f).astype(np.float32)
        self.bias_tensor = bias_tensor
        # ── bias 처리 (TPULinear와 동일) ───────────────────────
        if bias_tensor is not None:
            if hasattr(bias_tensor, 'detach'):
                self.bias = (bias_tensor.detach().cpu().numpy().astype(np.float32)
                             / out_scale_f)
            else:
                self.bias = (np.asarray(bias_tensor, dtype=np.float32)
                             / out_scale_f)
        else:
            self.bias = np.zeros(self.out_C, dtype=np.float32)

        # ── 4분할 + param_buf (TPULinear와 동일) ───────────────
        w_slices = np.vsplit(w_np, 4)
        m_slices = np.split(m_scale_per_channel, 4)
        b_slices = np.split(self.bias, 4)

        self.src2_list      = []
        self.src2_c_list    = []
        self.param_buf_list = []

        for w_s, m_s, b_s in zip(w_slices, m_slices, b_slices):
            current_rows = w_s.shape[0]
            remainder = current_rows % 16
            if remainder != 0:
                pad = 16 - remainder
                w_s = np.pad(w_s, ((0, pad), (0, 0)), mode='constant', constant_values=0)
                m_s = np.pad(m_s, (0, pad), mode='constant')
                b_s = np.pad(b_s, (0, pad), mode='constant')
                print(f"Padding added: {current_rows} -> {w_s.shape[0]}")

            w_s_tensor = torch.from_numpy(w_s).to(torch.int8)
            W_proc, W_c = preprocess_weight_for_tpu(w_s_tensor)

            s2_c = allocate(shape=W_c.shape, dtype=np.int8)
            s2_c[:] = W_c
            self.src2_list.append(W_proc.shape)
            self.src2_c_list.append(s2_c)

            num_ch = w_s.shape[0]
            interleaved = np.empty(num_ch * 2, dtype=np.float32)
            interleaved[0::2] = m_s
            interleaved[1::2] = b_s
            param_buf = allocate(shape=(num_ch * 2,), dtype=np.float32)
            param_buf[:] = interleaved
            param_buf.flush()
            self.param_buf_list.append(param_buf)

        self.m_scale_all = np.concatenate([
            np.asarray(pb)[0::2] for pb in self.param_buf_list
        ]).astype(np.float32)                                  # [768]

        self.bias_all = np.concatenate([
            np.asarray(pb)[1::2] for pb in self.param_buf_list
        ]).astype(np.float32)                                  # [768]

        self.weight_T = self.weight.astype(np.int32).T.copy()
        # ── 결과/입력 버퍼 ─────────────────────────────────────
        self.result_buf   = np.empty((self.M, self.out_C), dtype=np.int8)
        self.result_torch = torch.from_numpy(self.result_buf)
        self.patch_buf    = np.zeros((self.M, self.K), dtype=np.uint8)
        self.patches_buf = np.empty(
            (self.hw.batch_size * self.N, self.K), dtype=np.uint8
        )

    def forward(self, x):
        """
        x: quint8 tensor [B, 3, 224, 224]
        return: quint8 tensor [B, 196, 768]  (TPULinear와 동일하게 quint8 반환)
        """
        t0 = time.perf_counter()
        B     = x.shape[0]
        in_zp = x.q_zero_point()

        # ① int_repr
        ta = time.perf_counter()
        x_np = x.int_repr().cpu().numpy()
        # print(f"① int_repr: {(time.perf_counter()-ta)*1000:.2f}ms")

        # ② im2col
        ta = time.perf_counter()
        patches = self._im2col(x_np)
        patches = patches.reshape(B * self.N, self.K)
        # print(f"② im2col: {(time.perf_counter()-ta)*1000:.2f}ms")

        # ③ 버퍼 복사
        ta = time.perf_counter()
        self.patch_buf[:self.M, :] = patches
        self.patch_buf[self.M:, :] = 0
        # print(f"③ buf copy: {(time.perf_counter()-ta)*1000:.2f}ms")

        # ④ memmove
        ta = time.perf_counter()
        ctypes.memmove(
            self.hw.ip_buf_act.ctypes.data,
            self.patch_buf.ctypes.data,
            self.patch_buf.nbytes
        )
        # print(f"④ memmove: {(time.perf_counter()-ta)*1000:.2f}ms")

        # ⑤ flush
        ta = time.perf_counter()
        self.hw.ip_buf_act.flush()
        # print(f"⑤ flush: {(time.perf_counter()-ta)*1000:.2f}ms")

        # ── ④ 인터럽트 준비 ────────────────────────────────────
        Interrupt_write(self.INTERRUPT1)
        irq_future = asyncio.run_coroutine_threadsafe(
            anext(interrupt_monitor(self.INTERRUPT1, num_events=4)),
            self.hw.irq_loop
        )

        # ── ⑤ TPU GEMM 실행 ────────────────────────────────────
        t1 = time.perf_counter()
        for i in range(4):
            tpu_node = getattr(self.hw.ip_ol, f'TPU_PROCESSOR_{i}')
            run_sa(
                tpu_node,
                patches,
                self.hw.ip_buf_act.device_address,
                self.src2_list[i],
                self.src2_c_list[i],
                self.hw.ip_buf_dst[i],
                self.param_buf_list[i],
                int(in_zp),
                int(self.out_zp)
            )

        # ── ⑥ 인터럽트 대기 ────────────────────────────────────
        status = irq_future.result(timeout=5000)
        t2 = time.perf_counter()

        if status is None:
            read_value = self.INTERRUPT1.read(0x00)
            raise RuntimeError(f"TPU Timeout! status: {hex(read_value)}")

        # ── ⑦ 결과 수집 (TPULinear와 동일) ─────────────────────
        col_size = self.out_C // 4
        for i, d in enumerate(self.hw.ip_buf_dst):
            arr = np.asarray(d).ravel()
            arr = arr[:self.M * col_size].reshape(self.M, col_size)
            self.result_buf[:self.M, i*col_size:(i+1)*col_size] = arr[:self.M, :col_size]

        # ── ⑧ quint8 반환 (TPULinear와 동일) ───────────────────
        res_torch = self.result_torch[:self.M, :self.out_C].reshape(B, self.N, self.out_C)
        out_np = res_torch.numpy().transpose(0, 2, 1).reshape(B, self.out_C, 14, 14)

        out_int   = torch.from_numpy(out_np.copy()).to(torch.uint8)
        out_quant = torch._make_per_tensor_quantized_tensor(
            out_int,
            scale      = float(self.out_scale),
            zero_point = int(self.out_zp)
        )

        t3 = time.perf_counter()
        # print(f"PatchEmbed: total={(t3-t2)*1000:.1f}ms | gemm={(t2-t1)*1000:.1f}ms | preprocess : total={(t1-t0)*1000:.1f}ms")

        ########################################### method2 ####################################################
        #patches_i32 = patches.astype(np.int32) - int(in_zp)      # [B, 196, 768]
        #cpu_gemm = patches_i32 @ self.weight_T   # [B, 196, 768]
        #cpu_fp = cpu_gemm.astype(np.float32) * self.m_scale_all + self.bias_all  # [B, 196, 768]
        #cpu_q = np.clip(
        #    np.round(cpu_fp) + self.out_zp, 0, 255
        #).astype(np.uint8)                                        # [B, 196, 768]
        #cpu_q_np = cpu_q.reshape(B,self.N,768).transpose(0, 2, 1) .reshape(B, self.out_C, 14, 14)           # [B, 768, 14, 14]
        #out_quant = torch._make_per_tensor_quantized_tensor(
        #    torch.from_numpy(cpu_q_np.copy()),
        #    scale      = float(self.out_scale),
        #    zero_point = int(self.out_zp)
        #)

        ######################################## method3 ##############################################
        #x_dq = x.dequantize()
        #w_dq = self.weight_ori.dequantize()
        #cpu_out = torch.nn.functional.conv2d(
        #    x_dq,
        #    w_dq,
        #    bias   = None,                            # bias는 나중에 따로
        #    stride = 16,
        #    padding= 0
        #)

        #if self.bias_tensor is not None:
        #    if isinstance(self.bias_tensor, np.ndarray):
        #        bias_dq = torch.from_numpy(self.bias_tensor).float()
        #    else:
        #        bias_dq = self.bias_tensor.detach().cpu().float()

        #    cpu_out = cpu_out + bias_dq.reshape(1, -1, 1, 1)   # view → reshape
        # [B, 768, 14, 14] 그대로 반환
        # reshape → permute → cat 은 graph가 그대로 처리
        #out_quant = torch.quantize_per_tensor(
        #    cpu_out,
        #    scale      = float(self.out_scale),
        #    zero_point = int(self.out_zp),
        #    dtype      = torch.quint8
        #)

        # TPU 결과와 비교
        '''
        tpu_q = out_int.numpy().reshape(self.N, self.out_C)          # [196, 768]
        diff  = np.abs(tpu_q.astype(np.int32) - cpu_q.astype(np.int32))
        wrong = np.where(diff > 1)

        if len(wrong[0]) != 0:
            print(f"❌ wrong detected: {len(wrong[0])}개")
            print(f"  tpu: {tpu_q[wrong][:8]}")
            print(f"  cpu: {cpu_q[wrong][:8]}")
            print(f"  diff max: {diff.max()}")
            breakpoint()
        else:
            print("✅ PatchEmbed correct")
        '''
        return out_quant   # [B, 196, 768] quint8

    @staticmethod
    def _im2col(x_np):
        B, C, H, W = x_np.shape
        P     = 16
        H_out = H // P
        W_out = W // P
        x = x_np.reshape(B, C, H_out, P, W_out, P)   # [B, C, 14, 16, 14, 16]
        x = x.transpose(0, 2, 4, 1, 3, 5)             # [B, 14, 14, C, 16, 16]
        x = x.reshape(B, H_out * W_out, C * P * P)    # [B, 196, 768]
        return np.ascontiguousarray(x)


# ── transform 함수도 x_scale 전달하도록 수정 ───────────────────
def transform_conv_to_tpu(model, hw):
    graph = model.graph
    nodes_to_replace_conv = []

    for k in graph.nodes:
        if k.op == 'call_module' and k.name == 'conv_proj':
            try:
                submod = model.get_submodule(k.target)
                if isinstance(submod, torch.nn.quantized.Conv2d):
                    nodes_to_replace_conv.append(k)
                    print(f"Conv 발견: {k.target} | {submod}")
            except AttributeError:
                pass
    for k in nodes_to_replace_conv:
        submod = model.get_submodule(k.target)
        # quantize_per_tensor 노드에서 x_scale 추출
        qpt_node = k.args[0]
        x_scale  = qpt_node.args[1]   # quantize_per_tensor의 scale

        tpu_module = TPUPatchEmbedding(
            name          = k.target,
            x_scale       = getattr(model,x_scale.target),
            weight_tensor = submod.weight(),    # per-channel quant 정보 포함
            bias_tensor   = submod.bias(),
            out_scale     = submod.scale,
            out_zp        = submod.zero_point,
            hw            = hw
        )

        tpu_target = k.target + '_tpu'
        model.add_submodule(tpu_target, tpu_module)

        with graph.inserting_after(k):
            input_node = k.args[0]
            new_node   = graph.call_module(tpu_target, args=(input_node,))
        k.replace_all_uses_with(new_node)
        graph.erase_node(k)

    model.recompile()
    model.graph.lint()
    return model


from torch.fx import Interpreter

# 모델이 fx.GraphModule 형태라면 Interpreter를 사용해 한 단계씩 실행 가능
def check_precision(fx_module, example_input):
    interpreter = Interpreter(fx_module)
    # 실제 실행 중에 각 노드의 dtype을 가로채서 출력
    for node in fx_module.graph.nodes:
        # 이 시점에서 node.meta['tensor_meta'].dtype을 확인하면
        # 설계된 Precision(float32 혹은 int8)이 나옵니다.
        print(f"Node: {node.name}, Precision: {node.meta.get('tensor_meta')}")


def transform_mha_to_tpu(model, hw):
    graph = model.graph
    for layer_idx in range(12):
        qkv_target  = f'encoder.layers.encoder_layer_{layer_idx}.self_attention.qkv'
        proj_target = f'encoder.layers.encoder_layer_{layer_idx}.self_attention.proj'
        attn_drop_target = f'encoder.layers.encoder_layer_{layer_idx}.self_attention.attn_drop'
        # ── 노드 찾기 ──────────────────────────────
        qkv_node  = None
        proj_node = None
        for node in graph.nodes:
            if node.op == 'call_module':
                if node.target == qkv_target:
                    qkv_node  = node
                if node.target == proj_target:
                    proj_node = node

        if qkv_node is None or proj_node is None:
            continue
        # ── MHA 입력 = qkv의 입력 ─────────────────
        mha_input = qkv_node.args[0]
        proj_input = proj_node.args[0]
        energy_layer = proj_node.args[0].args[0].args[0].args[0].args[0].args[0].args[0].args[0].args[0].args[0]
        attention_output_layer = proj_node.args[0].args[0].args[0].args[0]
        attention_input_layer = proj_node.args[0].args[0].args[0].args[0].args[0].args[0]
        # ── TPUMultiHeadAttention 생성 ────────────
        qkv_mod       = model.get_submodule(qkv_target)
        proj_mod      = model.get_submodule(proj_target)
        tpu_mha = TPUMultiHeadAttention(
            qkv_module  = qkv_mod,
            proj_module = proj_mod,
            qkv_act_scale = model.get_submodule(mha_input.target).scale,
            qkv_input_act_zero = model.get_submodule(mha_input.target).zero_point,

            proj_act_scale = find_output_scale(proj_node.args[0].args[0].args[0].args[0],model)[0],
            energy_scale=model.get_buffer(energy_layer.args[2].target),
            energy_zero=model.get_buffer(energy_layer.args[3].target),
            attention_input_scale = model.get_buffer(attention_input_layer.args[1].target) ,
            attention_input_zero = model.get_buffer(attention_input_layer.args[2].target),
            attention_output_scale=model.get_buffer(attention_output_layer.args[2].target),
            attention_output_zero=model.get_buffer(attention_output_layer.args[3].target),
            num_heads   = 12,
            hw          = hw
        )

        tpu_name = f'tpu_mha_{layer_idx}'
        setattr(model, tpu_name, tpu_mha)

        # ── qkv ~ proj 사이 노드 수집 ────────────
        nodes_between = []
        collecting = False
        for node in graph.nodes:
            if node == qkv_node:
                collecting = True
            if collecting:
                nodes_between.append(node)
            if node == proj_node:
                break

        # ── 새 노드 삽입 ──────────────────────────
        with graph.inserting_after(proj_node):
            new_node = graph.call_module(
                tpu_name,
                args=(mha_input,)
            )
            proj_node.replace_all_uses_with(new_node)

        # ── 중간 노드 역순으로 제거 ───────────────
        for node in reversed(nodes_between):
            if len(node.users) == 0:
                graph.erase_node(node)

    graph.lint()
    model.recompile()
    return model
import threading
from concurrent.futures import ThreadPoolExecutor
import math
import io
import onnxruntime as ort

# 클래스 밖에 선언

#@torch.jit.script
def softmax_requantize(t_in: torch.Tensor,
                       p_zp_f: float,
                       combined_scale: float,
                        valid_mask_scaled: torch.Tensor,) -> torch.Tensor:
    #t_in[:, :, 197:] = float('-inf')
    #t_in.sub_(p_zp_f)
    t_in.mul_(combined_scale)
    #index over 197 are making error, mask placed in wrong place.
    t_in = torch.softmax(t_in, dim=-1)
    t_in.mul_(valid_mask_scaled)
    t_in = torch.round(t_in)
    t_in = torch.clamp(t_in, 0, 255)
    t_in.sub_(128)
    return t_in

class TPUMultiHeadAttention(nn.Module):

    MAX_OUT_FEATURES = 208

    def __init__(self, qkv_module, proj_module,
                 qkv_act_scale,qkv_input_act_zero,proj_act_scale,
                 energy_scale, energy_zero,attention_input_scale, attention_input_zero, attention_output_scale, attention_output_zero, num_heads,  hw):
        super().__init__()
        from pynq import Interrupt
        self.INTERRUPT1 = hw.ip_ol.axi_intc_0
        self.tpu_irq = Interrupt('TPU_PROCESSOR_3/interrupt')
        self.num_heads  = num_heads
        self.qkv        = qkv_module
        self.proj       = proj_module
        self.hw         = hw
        self.original_row_nums = 197
        self.energy_scale = energy_scale
        self.energy_zero = energy_zero
        # __init__에서
        self.d_k = qkv_module._packed_params._weight_bias()[0].shape[0] // 3 // num_heads
        self.q_concat_memory = np.full( (self.hw.batch_size, 208, 768),  self.qkv.zero_point, dtype = np.uint8 )
        self._q_concat_torch = torch.from_numpy(self.q_concat_memory)
        self.k_concat_memory = np.full( (self.hw.batch_size, 208, 768),  self.qkv.zero_point, dtype = np.uint8 )
        self._k_concat_torch = torch.from_numpy(self.k_concat_memory)
        self.v_concat_memory = np.full( (self.hw.batch_size, 208, 768),  self.qkv.zero_point, dtype = np.uint8 )
        self._v_concat_torch = torch.from_numpy(self.v_concat_memory)

        self._head_pool = ThreadPoolExecutor(
            max_workers=3
        )

        self._attn_scale = float(1.0 / math.sqrt(self.d_k))
        '''
        self.qk_result_memory = np.empty(
            (B * self.num_heads, 208, 208), dtype=np.uint8)
        self._qk_result_torch = torch.from_numpy(self.qk_result_memory)

        # PV 결과: [B, heads, N, d_k] = [2, 12, 208, 64]
        self.pv_result_memory = np.empty(
            (B * self.num_heads, 208, 64), dtype=np.uint8)
        self._pv_result_torch = torch.from_numpy(self.pv_result_memory)
        '''

        if not hasattr(hw, 'irq_loop'):
            new_loop = asyncio.new_event_loop()
            hw.irq_loop = new_loop
            t = threading.Thread(target=start_irq_loop,
                                 args=(new_loop,), daemon=True)
            t.start()

        # QKV preprocess
        k_module, q_module, v_module = self._reorder_qkv_to_kqv(qkv_module)

        self.k_src2_list, self.k_src2_c_list, self.k_param_buf_list = self._preprocess_weight(k_module, qkv_act_scale)
        self.q_src2_list, self.q_src2_c_list, self.q_param_buf_list = self._preprocess_weight(q_module, qkv_act_scale)
        self.v_src2_list, self.v_src2_c_list, self.v_param_buf_list = self._preprocess_weight(v_module, qkv_act_scale)
        self.k_out_features= k_module._packed_params._weight_bias()[0].shape[0]
        self.q_out_features= q_module._packed_params._weight_bias()[0].shape[0]
        self.v_out_features= v_module._packed_params._weight_bias()[0].shape[0]
        self.k_shape = np.empty((self.hw.batch_size,12,64,208),dtype = np.int8)
        # PROJ preprocess
        self.proj_src2_list, self.proj_src2_c_list, self.proj_param_buf_list = self._preprocess_weight(proj_module, proj_act_scale)
        self.proj_out_features = proj_module._packed_params._weight_bias()[0].shape[0]

        proj_num_rows         = self.hw.batch_size * 208
        self.proj_padded_rows = (proj_num_rows)
        self.proj_padded_input = np.zeros((self.proj_padded_rows, 768), dtype=np.int8)
        self.proj_result_buf  = np.empty((proj_num_rows, self.proj_out_features), dtype=np.uint8)
        self.proj_result_torch = torch.from_numpy(self.proj_result_buf)
        self.proj_col_size    = self.proj_out_features // 4
        self.proj_actual_elements = self.proj_padded_rows * self.proj_src2_list[0].shape[1]
        self.proj_zp_int      = int(self.proj.zero_point)

        self.mha_result_buf = np.empty(
                (208*self.hw.batch_size, 768), dtype=np.uint8
            )
        self.mha_result_torch = torch.from_numpy(self.mha_result_buf)
        self.mha_col_size = 768 // 4



        #Matmul preprocess
        self.combined_scale = float(self.energy_scale) * self._attn_scale
        self.attention_input_scale = attention_input_scale
        self.attention_input_zero = attention_input_zero
        self.attention_output_scale = attention_output_scale
        self.attention_output_zero = attention_output_zero

        self._preprocess_matmul_param(
            qkv_module = qkv_module,
            p_scale    = energy_scale,    # matmul_24 출력 scale
            p_zp       = energy_zero,
            v_scale    = float(qkv_module.scale),  # V scale = QKV scale
            attn_scale = attention_input_scale, # matmul_25 출력 scale
            attn_zp    = attention_input_zero,
            row_nums   = 208
        )

        #softmax_process
        self._softmax_first_run = True
        self.inv_out_scale    = 1.0 /(float(self.attention_input_scale))
        self._valid_mask_scaled = torch.zeros(4, 208, 208)
        #self._valid_mask_scaled[:, :197, :197] = self.inv_out_scale
        self._valid_mask_scaled[:, :, :] = self.inv_out_scale
        self.combined_scale = float(self.energy_scale) * self._attn_scale
        self.p_zp_f         = float(self.energy_zero)

        self.neg_zp = np.uint8((256 - self.qkv.zero_point) & 0xFF)
        self.v3_buf_u8 = np.empty((self.hw.batch_size, 208, 768), dtype=np.uint8)
        self.v3_buf_i8 = self.v3_buf_u8.view(np.int8)
        self.scale_128 = float( 128 * self.attention_input_scale  * self.qkv.scale / self.attention_output_scale)

    def _reorder_qkv_to_kqv(self, qkv_module):
        weight, bias = qkv_module._packed_params._weight_bias()
        w_np     = weight.int_repr().detach().cpu().numpy()
        w_scales = weight.q_per_channel_scales().detach().numpy()
        w_zp     = weight.q_per_channel_zero_points().detach().numpy()
        b_np     = bias.detach().cpu().numpy().astype(np.float32)

        out_features = w_np.shape[0]
        chunk        = out_features // 3

        Q_w = w_np[:chunk, :];   K_w = w_np[chunk:2*chunk, :];   V_w = w_np[2*chunk:, :]
        Q_b = b_np[:chunk];      K_b = b_np[chunk:2*chunk];      V_b = b_np[2*chunk:]
        Q_s = w_scales[:chunk];  K_s = w_scales[chunk:2*chunk];  V_s = w_scales[2*chunk:]
        Q_z = w_zp[:chunk];      K_z = w_zp[chunk:2*chunk];      V_z = w_zp[2*chunk:]

        def make_module(w, b, s, z):
            import copy
            w_tensor = torch._make_per_channel_quantized_tensor(
                torch.from_numpy(w).to(torch.int8),
                torch.from_numpy(s).double(),
                torch.from_numpy(z).int(),
                axis=0
            )
            b_tensor = torch.nn.Parameter(
                torch.from_numpy(b), requires_grad=False
            )
            module = torch.ao.nn.quantized.Linear(
                in_features  = w.shape[1],  # 768
                out_features = w.shape[0],  # 768 (K만)
            )
            module.scale       = qkv_module.scale
            module.zero_point  = qkv_module.zero_point
            module._packed_params._weight_bias = lambda: (w_tensor, b_tensor)

            return module
        k_module = make_module(K_w, K_b, K_s, K_z)
        q_module = make_module(Q_w, Q_b, Q_s, Q_z)
        v_module = make_module(V_w, V_b, V_s, V_z)

        return k_module, q_module, v_module

    def _preprocess_matmul_param(self, qkv_module, p_scale, p_zp, v_scale, attn_scale, attn_zp, row_nums):
        """
        Q@K^T (energy) 와 P@V (attention) 연산을 위한 param_buf 생성

        energy (Q@K^T):
          M_scale = Q_scale * K_scale / P_scale
          bias    = 0

        attention (P@V):
          M_scale = attn_scale * V_scale / out_scale
          bias    = 0
        """
        qkv_scale = float(qkv_module.scale)

        # ── Energy (Q@K^T) param ───────────────────
        energy_M_scale = (qkv_scale * qkv_scale) / float(self.energy_scale)
        interleaved_energy       = np.empty(row_nums * 2, dtype=np.float32)
        interleaved_energy[0::2] = energy_M_scale
        interleaved_energy[1::2] = 0.0

        self.MM_energy_param_buf_list = [
            allocate(shape=(row_nums * 2,), dtype=np.float32)
            for _ in range(4)
        ]

        for i in range(4):
            self.MM_energy_param_buf_list[i][:] = interleaved_energy
        self.MM_energy_param_buf_list[i].flush()

        # ── Attention (P@V) param ──────────────────
        attn_M_scale = (float(self.attention_input_scale) * float(v_scale)) / float(self.attention_output_scale)

        interleaved_attn       = np.empty(row_nums * 2, dtype=np.float32)
        interleaved_attn[0::2] = attn_M_scale
        #interleaved_attn[0::2] = interleaved_attn[0::2]*self.combined_scale
        interleaved_attn[1::2] = 0.0

        self.MM_attn_param_buf_all = allocate(
                shape=(self.hw.batch_size * self.num_heads * row_nums * 2,),
                dtype=np.float32
            )

        self.mm_attn_param_np = np.asarray(self.MM_attn_param_buf_all).reshape(
                self.hw.batch_size * self.num_heads, row_nums * 2
            )
        from collections import namedtuple

        PhysAddr = namedtuple('PhysAddr', ['device_address'])

        self.MM_attn_param_buf_list = [
                PhysAddr(device_address=
                    self.MM_attn_param_buf_all.device_address + idx * row_nums * 2 * 4)  # float32=4bytes
                for idx in range(self.hw.batch_size * self.num_heads)
            ]

        for i in range(self.hw.batch_size*12):
            self.mm_attn_param_np[i] = interleaved_attn #broadcasting
        self.MM_attn_param_buf_all.flush()



        # scale, zp 저장
        self.p_scale   = float(p_scale)
        self.p_zp      = int(p_zp)

    def _preprocess_weight (self, module, act_scale):
        weight, bias = module._packed_params._weight_bias()
        w_np     = weight.int_repr().detach().cpu().numpy()
        w_slices = np.vsplit(w_np, 4)

        m_scale = (act_scale * weight.q_per_channel_scales()
                   / module.scale)
        m_slices = np.split(m_scale.detach().numpy(), 4)

        bias_fused = (bias / module.scale).detach().cpu().numpy()
        b_slices   = np.split(bias_fused, 4)

        src2_list, src2_c_list, param_buf_list = [], [], []
        for w_s, m_s, b_s in zip(w_slices, m_slices, b_slices):
            current_rows = w_s.shape[0]
            remainder    = current_rows % 16
            if remainder != 0:
                padding_size = 16 - remainder
                w_s = np.pad(w_s, ((0, padding_size), (0, 0)),
                             mode='constant', constant_values=0)
                m_s = np.pad(m_s, (0, padding_size), mode='constant')
                b_s = np.pad(b_s, (0, padding_size), mode='constant')

            W, W_c = preprocess_weight_for_tpu(
                torch.from_numpy(w_s).to(torch.int8))

            s2   = allocate(shape=W.shape,   dtype=np.int8)
            s2_c = allocate(shape=W_c.shape, dtype=np.int8)
            s2[:]   = W
            s2_c[:] = W_c
            src2_list.append(s2)
            src2_c_list.append(s2_c)

            num_ch      = w_s.shape[0]
            interleaved = np.empty(num_ch * 2, dtype=np.float32)
            interleaved[0::2] = m_s
            interleaved[1::2] = b_s
            param_buf    = allocate(shape=(num_ch * 2,), dtype=np.float32)
            param_buf[:] = interleaved
            param_buf.flush()
            param_buf_list.append(param_buf)

        return src2_list, src2_c_list, param_buf_list


    def TPU_QKVLinear(self, x, mode,q_zero_point, DATA_COPY = False):
        t0=time.perf_counter()
        src2_list     = getattr(self, f'{mode}_src2_list')
        src2_c_list   = getattr(self, f'{mode}_src2_c_list')
        dst_list = getattr(self.hw,f'ip_{mode}buf_dst')
        param_buf_list = getattr(self, f'{mode}_param_buf_list')
        out_features  = getattr(self, f'{mode}_out_features')
        concat_memory = getattr(self, f'{mode}_concat_memory')
        concat_memory_torch = getattr(self, f'_{mode}_concat_torch')

        original_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        num_rows = x_2d.shape[0] #197
        in_features = x_2d.shape[1] #768 or 3072
        padded_rows = (num_rows + 15) // 16 * 16

        if DATA_COPY == True:
            flat_data = x_2d.int_repr().cpu().numpy().flatten()
            num_elements = flat_data.size
            self.hw.ip_buf_act.flat[:num_elements] = flat_data
            current_input = flat_data.reshape(num_rows,in_features)
            pad_amt = padded_rows - current_input.shape[0] # 200 - 197 = 3
            current_input = np.pad(current_input, ((0, pad_amt), (0, 0)), mode='constant', constant_values=0)
        # 인터럽트 감시 시작
        t1=time.perf_counter()
        # print(f"QKV_LINEAR1 time = {(t1-t0)*1000:.1f}ms")
        t0=time.perf_counter()
        Interrupt_write(self.INTERRUPT1)
        for i in range(4):
            tpu_node = getattr(self.hw.ip_ol, f'TPU_PROCESSOR_{i}')
            run_sa(tpu_node,x_2d, self.hw.ip_buf_act.device_address, src2_list[i], src2_c_list[i], dst_list[i], param_buf_list[i],q_zero_point,self.qkv.zero_point)

        actual_results_elements = padded_rows * src2_list[0].shape[1]
        start_time = time.perf_counter()
        tile_row_nums = src2_list[0].shape[1]
        tile_col_nums = x.shape[1]
        B = x.shape[0]
        results = [None] * 4
        done_mask = 0
        target_mask = 0b1111

        start_time = time.perf_counter()

        while done_mask != target_mask:
            if (time.perf_counter() - start_time) > 5.0:
                read_value = self.INTERRUPT1.read(0x00)
                print(f"TPU Timeout! done={bin(done_mask)} reg={hex(read_value)}")
                breakpoint()
                raise RuntimeError(f"TPU Timeout! done={bin(done_mask)}")

            reg_val = self.INTERRUPT1.read(0x00)
            for i in range(4):
                bit = (1 << i)
                if (reg_val & bit) and not (done_mask & bit):
                    buf = dst_list[i]
                    arr = np.asarray(buf).reshape(-1)[:actual_results_elements].reshape(padded_rows, tile_row_nums)
                    needed = arr[:num_rows, :tile_row_nums]            # non-contig view
                    col_start = i * tile_row_nums
                    col_end   = col_start + tile_row_nums
                    for b in range(B):
                        np.copyto(
                            concat_memory[b, :self.original_row_nums, col_start:col_end],
                            needed[(b*padded_rows//B):(b*padded_rows//B)+self.original_row_nums, :]
                        )
                    done_mask |= bit
                else:
                    time.sleep(0.00005)
                    i=3
        t1=time.perf_counter()
        # print(f"QKV_LINEAR2 time = {(t1-t0)*1000:.1f}ms")
        t0=time.perf_counter()
        self.INTERRUPT1.write(0x0C, 0xF)

        out_quant = torch._make_per_tensor_quantized_tensor(
            concat_memory_torch,
            scale      = float(self.qkv.scale),
            zero_point = int(self.qkv.zero_point)
        )
        t1=time.perf_counter()
        # print(f"QKV_LINEAR3 time = {(t1-t0)*1000:.1f}ms")
        return out_quant

    def TPU_PROJLinear(self, x):
        t0=time.perf_counter()

        if x.dim() == 4:
            x = x.transpose(1, 2)
            x = x.reshape(x.shape[0], x.shape[1], -1)

        x_2d     = x.reshape(-1, x.shape[-1])
        num_rows = x_2d.shape[0]
        padded_rows = (num_rows + 15) // 16 * 16
        # 1. ravel (no copy)
        flat_data    = x_2d.int_repr().cpu().numpy().ravel()
        num_elements = flat_data.size
        import ctypes
        # 2. ctypes memmove (fastest)
        ctypes.memmove(
            self.hw.ip_buf_act.ctypes.data,
            flat_data.ctypes.data,
            flat_data.nbytes
        )

        # 3. pre-allocated padded input (no np.pad)
        current_input = self.proj_padded_input

        # 인터럽트 감시 시작
        Interrupt_write(self.INTERRUPT1)
        for i in range(4):
            tpu_node = getattr(self.hw.ip_ol, f'TPU_PROCESSOR_{i}')
            run_sa(tpu_node, current_input, self.hw.ip_buf_act.device_address, self.proj_src2_list[i], self.proj_src2_c_list[i], self.hw.ip_buf_dst[i],self.proj_param_buf_list[i],x.q_zero_point(),self.proj.zero_point)


        results = [None] * 4
        done_mask = 0
        target_mask = 0b1111
        actual_results_elements = padded_rows * self.proj_src2_list[0].shape[1]

        start_time = time.perf_counter()

        while done_mask != target_mask:
            if (time.perf_counter() - start_time) > 5.0:
                read_value = self.INTERRUPT1.read(0x00)
                print(f"TPU Timeout! done={bin(done_mask)} reg={hex(read_value)}")
                breakpoint()
                raise RuntimeError(f"TPU Timeout! done={bin(done_mask)}")

            reg_val = self.INTERRUPT1.read(0x00)
            new_bits = reg_val & ~done_mask

            if new_bits:
                for i in range(4):
                    if new_bits & (1 << i):
                        buf = self.hw.ip_buf_dst[i]
                        arr = (np.asarray(buf).reshape(-1)[:actual_results_elements].reshape(padded_rows, self.proj_src2_list[i].shape[1]))
                        self.mha_result_buf[:num_rows, i * self.mha_col_size:(i+1) * self.mha_col_size ] = arr[:num_rows, :self.mha_col_size]
                        #self.mha_result_buf[i] = arr[:num_rows, :(self.proj_out_features // 4)]
                done_mask |= new_bits
            else:
                time.sleep(0.00005)

        # 인터럽트 대기 (타임아웃 5초로 넉넉하게 설정)
        self.INTERRUPT1.write(0x0C, 0b1111)

        res_torch = self.mha_result_torch[:num_rows, :self.proj_out_features].reshape(x.shape[:-1] + (self.proj_out_features,))
        out_quant = torch._make_per_tensor_quantized_tensor(
            res_torch,
            scale      = float(self.proj.scale),
            zero_point = int(self.proj.zero_point)
        )
        t1=time.perf_counter()
        # print(f"PROJ_LINEAR time = {(t1-t0)*1000:.1f}ms")

        return out_quant

    def TPU_Matmul(self, a_shape, b_shape, a_zero_point, mode='QK'):
        """
        mode: 'QK' = Q@K^T → dequant + scale + softmax + 재양자화까지 처리
              'PV' = P@V → 결과 수집만

        QK mode:
            - TPU matmul 후, 4 head를 thread pool로 병렬 처리
            - head별 fused pipeline: read → dequant → ×(1/√d_k) → softmax → quantize
            - 결과는 self.hw.ip_buf_mm_P_list (PV 입력)와 qk_result_memory에 write
            - return: softmax + quantize된 quantized tensor (attention_input scale/zp)

        PV mode:
            - TPU matmul 후 결과만 copy
            - return: attention_output scale/zp로 wrap된 quantized tensor
        """
        # ─────────────────────────────────────────────
        # 0) Shape 및 공통 변수
        # ─────────────────────────────────────────────
        t00 = time.perf_counter()
        B, heads, M, K = a_shape.shape
        N = b_shape.shape[-1]
        padded_rows = M
        cols   = b_shape.shape[3]
        n_elem = padded_rows * cols
        # buf_list: mode에 따라 다르지만 group과는 무관 (루프 밖에서 한 번만 결정)
        if mode == 'QK':
            buf_list = self.hw.ip_buf_mm_OCM_list
        elif mode == 'PV':
            buf_list = self.hw.ip_buf_mm_Q_list
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # ─────────────────────────────────────────────
        # 1) QK mode 양자화 파라미터 미리 계산
        # ─────────────────────────────────────────────
        if mode == 'QK':
            # Dequant scale × attention scale (1/√d_k)을 곱셈 1번으로 합침

            # ───── head별 처리 함수 (QK용, closure로 위 변수 캡쳐) ─────

            def process_qk_head(group_):
                t0 = time.perf_counter()
                selected_group = group_%8
                # ─ 2) numpy → torch 변환 ─
                ############################# softmax method1 ##########################################

                BRAM_BASE  = 0xB000_0000
                H, W = 208, 208
                group_size = H * W

                b       = group_ // 12
                h_start = group_ % 12
                src_addr = self.hw.ip_buf_mm_OCM_list[selected_group]
                dst_addr = self.hw.ip_buf_mm_P_list[group_]
                if self._softmax_first_run:
                    # 첫번째는 그냥 바로 실행 (이전 작업 없음)
                    run_softmax(
                        self.hw.ip_ol.softmax_module_0,
                        dst           = dst_addr,
                        src           = src_addr,
                        height        = H*4,
                        width         = W,
                        #scale         = self.combined_scale,
                        scale = struct.unpack('<I', struct.pack('<f', self.inv_out_scale))[0],
                        softmax_scale = struct.unpack('<I', struct.pack('<f', self.combined_scale))[0],
                        poll          = False,   # non-blocking
                    )
                    self._softmax_first_run = False
                else:
                    run_softmax(
                        self.hw.ip_ol.softmax_module_0,
                        dst           = dst_addr,
                        src           = src_addr,
                        height        = H * 4,
                        width         = W,
                        #scale         = float(self.combined_scale),
                        scale = struct.unpack('<I', struct.pack('<f', self.inv_out_scale))[0],
                        softmax_scale = struct.unpack('<I', struct.pack('<f', self.combined_scale))[0],
                        poll          = False,   # non-blocking
                    )


                ############################## softmax method 2 #########################################
                # 이미 BRAM을 직접 바라보는 view
                #for i in range(4):
                #    self.hw._softmax_scratch_f32_torch[i].copy_(self.hw.ocm_u8_torch[selected_group + i]  )

                #t_in = self.hw._softmax_scratch_f32_torch
                #t1 = time.perf_counter()
                #t_in = softmax_requantize(
                #    t_in,
                #    0.0,
                #    float(self.combined_scale),
                #    self._valid_mask_scaled
                #)

                # ─ 6) 두 곳에 write ─
                #b       = group_ // 12
                #h_start = group_ % 12
                #ref = t_in.to(torch.int8).cpu().numpy()
                #wrong = np.where ( self.hw.P_strided[b, h_start:h_start+4] != ref )
                #if len( wrong[0] ) >2:
                #    breakpoint()
                #self.hw.P_strided[b, h_start:h_start+4].copy_(t_in.to(torch.int8).view(torch.uint8))

                #############################################################################################

        else:  # PV mode: 단순 copy만
            def process_pv_head(group_):
                np.copyto(
                    self.hw.pv_result_memory[group_:group_+4, :208, :self.hw.d_k],
                    self.hw.q_strided[group_:group_+4, :208, :self.hw.d_k]
                )

        # ─────────────────────────────────────────────
        # 2) Group별 루프 (TPU 4개 동시 실행 + head 4개 병렬 후처리)
        # ─────────────────────────────────────────────

        futures = []
        prev_group  = None
        self.tpu_nodes = [
            getattr(self.hw.ip_ol, f'TPU_PROCESSOR_{i}')
            for i in range(4)
        ]
        q_addrs = [self.hw.ip_buf_mm_Q_list[h].device_address
           for h in range(B * heads)]

        # TPU_Matmul 시작 전에
        a_zp_int      = int(a_zero_point)
        energy_zp_int = int(self.energy_zero)
        attn_zp_int   = int(self.attention_output_zero)
        for group in range(0, B * heads, 4):
            # ── (a) TPU 4개 동시 launch ──
            t0 = time.perf_counter()
            if mode == 'PV':
                b       = group // 12
                h_start = group % 12
            Interrupt_write(self.INTERRUPT1)
            irq_future = asyncio.run_coroutine_threadsafe(
                anext(interrupt_monitor(self.INTERRUPT1, num_events=4)),
                self.hw.irq_loop
            )

            for i in range(4):
                head_idx = group + i
                dst_idx = head_idx % 8
                if mode == 'QK':
                    debug=run_sa(self.tpu_nodes[i],
                           a_shape[0][0],
                           q_addrs[head_idx],
                           b_shape[0][0],
                           self.hw.ip_buf_mm_KT_list[head_idx],
                           self.hw.ip_buf_mm_OCM_list[dst_idx],
                           self.MM_energy_param_buf_list[i],
                           a_zp_int, int(self.energy_zero) )
                else:  # PV
                    # ── 2) correction → MM_attn bias 버퍼에 삽입 ──
                    run_sa(self.tpu_nodes[i],
                           a_shape[0][0],
                           self.hw.ip_buf_mm_P_list[head_idx].device_address,
                           b_shape[0][0],
                           self.hw.ip_buf_mm_V_list[head_idx],
                           self.hw.ip_buf_mm_Q_list[head_idx],
                           self.MM_attn_param_buf_list[head_idx],
                           a_zp_int, attn_zp_int)
            # ── (c) 4-head 병렬 처리 (thread pool) ──
            if prev_group is not None:
                if mode == 'QK':
                    process_qk_head(prev_group)
                else:
                    process_pv_head(prev_group)
            else:
                self.hw._softmax_scratch_f32_torch.zero_()

            status = irq_future.result(timeout=5000)
            while (self.hw.ip_ol.softmax_module_0.read(0x50) >> 31) & 0x1:
                time.sleep(0.0001)
            t1 = time.perf_counter()
            # print(f"softmax loop{group} time = {(t1-t0)*1000:.1f}ms")


            if status is None:
                breakpoint()
                raise RuntimeError("TPU Timeout!")
            prev_group     = group

        if prev_group is not None:
            if mode == 'QK':
                process_qk_head(prev_group)
            else:
                process_pv_head(prev_group)

        t11 = time.perf_counter()
        # print(f"Total_MATMUL({mode}) time = {(t11-t00)*1000:.1f}ms")

        t0 = time.perf_counter()


        # ─────────────────────────────────────────────
        # 3) 결과 반환 (zero copy view wrap)
        # ─────────────────────────────────────────────
        if mode == 'QK':
            # softmax + quantize된 결과 → PV 입력 형식 (attention_input scale/zp)
            out_quant = self.hw.P_strided
            #out_quant = torch._make_per_tensor_quantized_tensor(
            #    self.hw.P_strided,
            #    scale      = float(self.attention_input_scale),
            #    zero_point = int(self.attention_input_zero)
            #)
        else:  # PV
            out_quant = torch._make_per_tensor_quantized_tensor(
                self.hw._pv_result_view,
                scale      = float(self.attention_output_scale),
                zero_point = int(self.attention_output_zero)
            )
        t1 = time.perf_counter()
        # print(f"softmax loop last time = {(t1-t0)*1000:.1f}ms")
        return out_quant




    def preprocess_k(self, k_raw, x_shape):
        import ctypes
        B, N, C = x_shape
        N_pad = (N + 15) // 16 * 16
        timings = {}

        # ① dtype 변환
        t = time.perf_counter()
        #k3 = self._k_concat_torch[:, :N, :].to(torch.int16)
        #k3.sub_(self.qkv.zero_point)
        #k3 = k3.to(torch.int8)  # (B, N, heads*d_k)
        neg_zp = np.uint8((256 - self.qkv.zero_point) & 0xFF)
        k3_np = (self.k_concat_memory[:, :N, :] + neg_zp).view(np.int8)
        k3 = torch.from_numpy(k3_np)
        timings['dtype'] = (time.perf_counter() - t) * 1000

        # ② padding
        t = time.perf_counter()
        if N != N_pad:
            k3 = torch.nn.functional.pad(k3, (0, 0, 0, N_pad - N))
        timings['pad'] = (time.perf_counter() - t) * 1000

        # ③ 한번에 reshape/permute (loop 제거)
        t = time.perf_counter()
        k3 = k3.reshape(B, N_pad, self.num_heads, self.d_k)   # (B, N_pad, heads, d_k)
        k3 = k3.permute(0, 2, 1, 3)                            # (B, heads, N_pad, d_k)
        k3 = k3.reshape(B, self.num_heads, N_pad//16, 16, self.d_k)
        #k3 = k3.permute(0, 1, 2, 4, 3).contiguous()           # (B, heads, N_pad//16, d_k, 16)
        k_np = k3.numpy()
        np.copyto(   self.hw._KT_scratch,   k_np.transpose(0, 1, 2, 4, 3) ) # 2,12,13,64,16

        timings['reshape'] = (time.perf_counter() - t) * 1000

        # ⑤ memmove loop
        t = time.perf_counter()

        # 한번에 복사 (padding 위치는 건드리지 않음)
        np.copyto(
            self.hw.kt_strided,
            self.hw._KT_scratch.reshape(self.hw.slots, self.hw.slot)
        )

        self.hw.ip_buf_mm_KT_all.flush()

        timings['memmove'] = (time.perf_counter() - t) * 1000

        total = sum(timings.values())
        #print(f"preprocess_k {total:.1f}ms | " +
        #      " ".join(f"{k}={v:.1f}" for k, v in timings.items()))

        return k3
    def preprocess_q(self, q_raw, x_shape):
        t0 = time.perf_counter()
        B, N, C = x_shape

        # ❌ 기존 문제들:
        # 1. q3.astype(np.uint8) → 이미 uint8인데 복사 발생
        # 2. torch.permute → non-contiguous tensor
        # 3. 이중 for loop → 느림
        # 4. q3[b,h].reshape(-1) → 매번 reshape

        # ✅ 최적화:
        # 1) zero-copy numpy view (이미 uint8)
        q_np = self.q_concat_memory[:, :N, :]          # [B, N, 768] view

        # 2) reshape → [B, N, 12, 64] view (복사 없음)
        q_np = q_np.reshape(B, N, self.hw.num_heads, self.d_k)

        q3 = q_np.transpose(0, 2, 1, 3)
        # ③ transpose + copy 한번에
        np.copyto(self.hw.q_strided, q3.reshape(self.hw.batch_size*12,208,64))
        self.hw.ip_buf_mm_Q_all.flush()
        t1 = time.perf_counter()
        #print(f"preprocess_q {(t1-t0)*1000:.2f}ms")
        return q3  # [B, 12, N, 64]

    def preprocess_v(self, v_raw, x_shape):
        t = time.perf_counter()
        B, N, C = x_shape
        N_pad = (N + 15) // 16 * 16
        timings = {}
        # ① dtype 변환
        t = time.perf_counter()
        np.add(self.v_concat_memory[:, :N, :], self.neg_zp,out=self.v3_buf_u8[:B, :N, :])   # ← 새 배열 할당 없음
        v3_np = self.v3_buf_i8[:B, :N, :]
        #v3_np = (self.v_concat_memory[:, :N, :] + neg_zp).view(np.int8)
        #print(f"  1.dtype:      {(time.perf_counter()-t)*1000:.1f}ms")
        t = time.perf_counter()

        # [B, N, 768] → [B, N, heads, d_k] → [B, heads, N, d_k//16, 16]
        t1=time.perf_counter()
        #print(f"MHA_3_2 time = {(t1-t0)*1000:.1f}ms")
        t0=time.perf_counter()
        v_for_sum = v3_np.reshape(self.hw.batch_size, N, self.hw.num_heads, 64) #(2,208,12,64)

        #v_sum = v_for_sum.astype(np.float32).sum(axis=1)
        v_sum = v_for_sum.sum(axis=1, dtype=np.int32)
        correction = (v_sum * self.scale_128).reshape(B*self.hw.num_heads,64)
        self.mm_attn_param_np[:,1::2][:,:64] =  correction
        #print(f"  2.correction: {(time.perf_counter()-t)*1000:.1f}ms")
        t = time.perf_counter()

        self.MM_attn_param_buf_all.flush()
        #print(f"  3.flush1:     {(time.perf_counter()-t)*1000:.1f}ms")
        t = time.perf_counter()

        if N != N_pad:
            v3_np = np.pad(v3_np, ((0,0),(0,N_pad-N),(0,0)))
        timings['pad'] = (time.perf_counter() - t) * 1000


        v_np = v3_np.reshape(B, N_pad, self.hw.num_heads, 64)

        # transpose [B,N,heads,d_k] → [B,heads,N,d_k//16,16]
        v_src = np.ascontiguousarray(
            v_np.transpose(0, 2, 1, 3)              # [2,12,208,64]
            .reshape(self.hw.batch_size, self.hw.num_heads, N_pad, 4, 16)  # [2,12,208,4,16]
            .transpose(0, 1, 3, 2, 4)              # [2,12,4,208,16]
        )

        np.copyto(self.hw.v_strided, v_src)
        #print(f"  3.flush1:     {(time.perf_counter()-t)*1000:.1f}ms")
        t = time.perf_counter()

        timings['reshape'] = (time.perf_counter() - t) * 1000
        self.hw.ip_buf_mm_V_all.flush()
        #print(f"  5.flush2:     {(time.perf_counter()-t)*1000:.1f}ms")
        t = time.perf_counter()

        return torch.from_numpy(v_np.transpose(0, 2, 1, 3))  # [B, heads, N, d_k]

    import ctypes
    def preprocess_k_wrapper(self,k_raw, x_shape):
        return self.preprocess_k(k_raw, x_shape)

    def preprocess_q_wrapper(self,q_raw, x_shape):
        return self.preprocess_q(q_raw, x_shape)

    def preprocess_v_wrapper(self,v_raw, x_shape):
        return self.preprocess_v(v_raw, x_shape)

    def forward(self, x):
        t_init = time.perf_counter()
        t0 = time.perf_counter()
        B, N, C = x.shape
        d_k = C // self.num_heads
        scale= d_k ** -0.5
        q_zero_point = x.q_zero_point()
        #qkv = self.qkv(x)
        #breakpoint()
        #qkv_f = qkv.dequantize()
        #qkv_f = qkv_f.reshape(B, N, 3, self.num_heads, d_k)
        #qkv_f = qkv_f.permute(2, 0, 3, 1, 4)
        #qq, k, v = qkv_f.unbind(0)
        #q,k,v: [B, heads, N, d_k]
        #v2 = v2.reshape(B, N, self.num_heads, d_k).permute(0, 2, 1, 3)

        if N%16 !=0:
            N=(N+15)//16 * 16
            pad_amt = N - x.shape[1] # 200 - 197 = 3
            x = x.int_repr()
            x = np.pad(x, ((0,0), (0, pad_amt), (0, 0)), mode='constant', constant_values=q_zero_point)

        # ── 1. QKV Linear (TPU) ───────────────────
        thread_results ={}
        import ctypes
        ctypes.memmove(
            self.hw.ip_buf_act.ctypes.data,  # dst
            x.ctypes.data,                     # src
            x.nbytes                           # size
        )
        t1=time.perf_counter()
        # print(f"acivation move time = {(t1-t0)*1000:.1f}ms")
        t0=time.perf_counter()

        k2 = self.TPU_QKVLinear(x, 'k',q_zero_point)
        kt_thread = self._head_pool.submit(self.preprocess_k_wrapper, k2, k2.shape)


        q2 = self.TPU_QKVLinear(x, 'q',q_zero_point)  # K preprocess와 overlap!
        qt_thread = self._head_pool.submit(self.preprocess_q_wrapper, q2, q2.shape)

        v2 = self.TPU_QKVLinear(x, 'v',q_zero_point)  # K,Q preprocess와 overlap!
        vt_thread = self._head_pool.submit(self.preprocess_v_wrapper, v2, v2.shape)

        k_shape=kt_thread.result()
        q_shape= qt_thread.result()
        # ── 3. Q @ K^T (CPU torch.matmul) ─────────
        #combined = torch.cat([q2,k2,v2] , dim=-1)
        #wrong = torch.where(qkv != combined[:, :197, :])
        #if len(wrong[0]) > 10:
        #    breakpoint()
        #k=k2.dequantize()
        #qq=q2.dequantize()
        #v=v2.dequantize()
        #k=k.reshape(B, 208, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        #qq=qq.reshape(B, 208, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        #v=v.reshape(B, 208, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        #attn2 = torch.matmul(qq, k.transpose(-2, -1))
        #attn2 = attn2.dequantize() * scale
        # ── 4. softmax (CPU) ──────────────────────
        #attn2 = attn2.softmax(dim=-1)
        #attn2 = torch.clamp( torch.round(attn2 / self.attention_input_scale) + self.attention_input_zero,
        #        0, 255
        #    ).to(torch.uint8)  # [B, heads, N, N]
        #attn2 = (attn2.to(torch.int16) - 128).to(torch.int8).view(torch.uint8)
        #self.hw.P_strided.copy_(
        #    (attn2)
        #)
        #self.hw.ip_buf_mm_P_all.flush()
        #t2 = time.perf_counter()
        t1=time.perf_counter()
        # print(f"QKV_Linear time = {(t1-t0)*1000:.1f}ms")
        t0=time.perf_counter()
        # --TPU Q@ K^T + SOFTMAX----------------------------------------
        attn = self.TPU_Matmul(q_shape,self.k_shape,q2.q_zero_point())
        #breakpoint()
        #wrong1 = np.where(attn2 != attn.int_repr())
        #if len(wrong1[0]) > 4:
        #    breakpoint()
        '''
#out = torch.from_numpy(q_shape[0][0].astype(np.int32) - self.qkv.zero_point) @ (k_shape[0][0]).to(torch.int32) * self.qkv.scale*self.qkv.scale/self.energy_scale +self.energy_zero

#self.hw.ip_buf_mm_OCM_list[0]


A=k2.int_repr().to(torch.int8)
B=q2.int_repr().to(torch.int8)
A  = A.reshape(2, k2.shape[1], self.num_heads, self.d_k).permute(0, 2, 1, 3)
B  = B.reshape(2, q2.shape[1], self.num_heads, self.d_k).permute(0, 2, 1, 3)
out = (B[1][8] - self.qkv.zero_point).to(torch.int32) @ (A[1][8] - self.qkv.zero_point).to(torch.int32).T * 0.00355
out=out*scale*self.p_scale
out = out.softmax(dim=-1)
out = torch.clamp( torch.round(out / self.attention_input_scale) + self.attention_input_zero, 0, 255 ).to(torch.uint8)
attn2[0][0].to(torch.int8)-self.p_zp)*self.p_scale
        '''
        t1=time.perf_counter()
        # print(f"Q*KT+SOFTMAX time = {(t1-t0)*1000:.1f}ms")
        t0=time.perf_counter()
        v_shape = vt_thread.result()
        # ── 5. P @ V (CPU torch.matmul) ───────────
        #x = torch.matmul(attn.dequantize(), v_shape*self.qkv.scale)  # [B, heads, N, d_k]
        #x = x.transpose(1, 2).contiguous()
        #x = x.reshape(B, N, C)    # [B, N, C]
        # ── 7. proj Linear (TPU) ──────────────────
        # proj 입력을 quantize
        #x = torch.quantize_per_tensor(     x,   scale      = float(self.attention_output_scale),     zero_point = int(self.attention_output_zero),        dtype      = torch.quint8     )


        #attn.dequantize()[0][0]@(v_shape*self.qkv.scale)[0][0]
        #(attn.int_repr()[0][0].to(torch.int32) - self.attention_input_zero) @ (v_shape[0][0] ).to(torch.int32) * self.attention_input_scale * self.qkv.scale  #
        #(attn.int_repr()[0][0].to(torch.int32) - self.energy_zero) @ (v_shape[0][0] ).to(torch.int32) * self.energy_scale * self.qkv.scale

        # P 버퍼에 저장
        #attn = torch.clamp( torch.round(attn / self.attention_input_scale) + self.attention_input_zero,
        #        0, 255
        #    ).to(torch.uint8)  # [B, heads, N, N]
        t1=time.perf_counter()
        # print(f"Q*KT+SOFTMAX+v_shape time = {(t1-t0)*1000:.1f}ms")
        t0=time.perf_counter()
        x2= self.TPU_Matmul(attn, v_shape, self.attention_input_zero, mode = 'PV')
        t1=time.perf_counter()
        # print(f"PV time = {(t1-t0)*1000:.1f}ms")
        t0=time.perf_counter()

        x2 = x2.transpose(1, 2).contiguous()
        x2 = x2.reshape(B, N, C)

        #wrong1 = np.where(x.int_repr() != x2.int_repr())
        #if len(wrong1[0]) > 4:
        #    for i in range(2):
        #        for j in range(12):
        #            wrong = np.where(x.int_repr()[i][j] != x2.int_repr()[i][j])
        #            if(len(wrong[0]) > 4):
        #                breakpoint()
        #x = x.dequantize()
        # ── 6. concat reshape (CPU) ───────────────
        #x = x.transpose(1, 2).contiguous()
        #x = x.reshape(B, N, C)    # [B, N, C]

        # ── 7. proj Linear (TPU) ──────────────────
        # proj 입력을 quantize
        #x2 = self.proj(x2)
        x = self.TPU_PROJLinear(x2)
        #wrong= np.where(x2 != x)
        #if(len(wrong[0])>10):
        #    breakpoint()

        x  = x[:, :197, :]
        t1=time.perf_counter()
        # print(f"CONCAT+PROJ time = {(t1-t0)*1000:.1f}ms")
        # print(f"MHA_TOTAL time = {(t1-t_init)*1000:.1f}ms")

        return x

def run_softmax(
    softmax_node,
    dst,
    src,
    height: int,
    width:  int,
    scale:  int  = 0x3F800000,   # FP32 1.0
    softmax_scale: int = 0x3E000000,
    start_val: int = 0x80000003,
    poll:  bool  = True,
):
    CSRA_CONTROL = 0x40
    CSRA_SCALE1  = 0x54
    CSRA_SCALE2  = 0x58
    CSRA_RDADDR  = 0x5c
    CSRA_WRADDR  = 0x60
    CSRA_MATRIX  = 0x50

    rd_phys = int(src.device_address)
    wr_phys = int(dst.device_address)

    # 1. CONTROL
    softmax_node.write(CSRA_CONTROL, start_val)
    # 2. SCALE
    softmax_node.write(CSRA_SCALE1,   scale & 0xFFFF_FFFF)
    softmax_node.write(CSRA_SCALE2,   softmax_scale & 0xFFFF_FFFF)
    # 3. RDADDR
    softmax_node.write(CSRA_RDADDR,  rd_phys)
    # 4. WRADDR
    softmax_node.write(CSRA_WRADDR,  wr_phys)
    # 5. MATRIX (GO → FSM 시작)
    matrix_val = (
        (1                   << 31) |
        ((height & 0xFFF)    << 16) |
        ( width  & 0xFFFF)
    )
    softmax_node.write(CSRA_MATRIX, matrix_val)

    # 6. DONE 폴링
    #if poll:
    #    while True:
    #        value = softmax_node.read(CSRA_MATRIX)
    #        if value & 0x2000_0000:   # bit29 = DONE
    #            break

    return {"height": height, "width": width,
            "scale": scale, "rd_phys": rd_phys, "wr_phys": wr_phys}


def run_sa(
    tpu_node,
    src_act,
    src1_1,
    src1_2,
    src1_2_CONCAT,
    dst1,
    src3_param,          # ← 추가: scale/bias 파라미터 주소 (source3)
    x_zp: int = 128,     # ← 추가: activation zero point
    out_zp: int = 0,
    relu: int = 0,
    sa_start_val: int = 0x80000000,
    timeout_s: float = 2.0,
    do_flush: bool = True,
    do_invalidate: bool = True,
    poll: bool = True,
    LITE = False
):
    CSRA_CONTROL   = 0x00
    SA_SOURCE1     = 0x04
    SA_SOURCE2     = 0x08
    SA_CONT1       = 0x0C
    SA_CONT2       = 0x10
    SA_DESTINATION = 0x14
    SA_Parameter1  = 0x18   # ← 추가: source3 (M_scale/bias 주소)
    SA_Parameter2  = 0x20

    M, K1 = src_act.shape
    if isinstance(src1_2, tuple):
        K2, N = src1_2
    else:
        K2, N = src1_2.shape
    #Md, Nd = dst1.shape

    if K1 != K2:
        raise ValueError(
            f"Shape mismatch: src1_1 is (M,K)=({M},{K1}) "
            f"but src1_2 is (K,N)=({K2},{N})"
        )
    K = K1
    src1_1_phys = int(src1_1)
    src1_2_phys = int(src1_2_CONCAT.device_address)
    dst1_phys = int(dst1.device_address)
    src3_phys     = int(src3_param.device_address)


    tpu_node.write(CSRA_CONTROL,sa_start_val)
    tpu_node.write(SA_DESTINATION, dst1_phys)
    tpu_node.write(SA_SOURCE1, src1_1_phys)
    tpu_node.write(SA_SOURCE2, src1_2_phys)
    if LITE == False:
        tpu_node.write(SA_CONT1, pack_cont1(relu, N, K))
        tpu_node.write(SA_CONT2, pack_cont2(relu, M))
        tpu_node.write(SA_Parameter2,  out_zp <<8 | x_zp & 0xFF)        # ← 추가: x_zp (8bit)
    tpu_node.write(SA_Parameter1,  src3_phys)
    tpu_node.write(CSRA_CONTROL, sa_start_val | 0x1)


    return {
        "M":           M,
        "K":           K,
        "N":           N,
        "src1_1_phys": src1_1_phys,
        "src1_2_phys": src1_2_phys,
        "dst1_phys":   dst1_phys,
        "src3_phys":   src3_phys,   # ← 추가
        "x_zp":        x_zp,        # ← 추가
    }



def start_irq_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


class TPULinear(nn.Module):
    def __init__(self,name, x_scale, weight_tensor, bias_tensor,out_scale,out_zp,hw):

        super().__init__()
        self.hw = hw
        self.name = name
        #self.out_scale = out_scale
        out_scale_float = float(out_scale)
        self.out_scale  = out_scale_float

        self.out_zp = out_zp
        self.INTERRUPT1 = hw.ip_ol.axi_intc_0
        if not hasattr(hw, 'irq_loop'):
            new_loop = asyncio.new_event_loop()
            hw.irq_loop = new_loop  # <--- 여기서 AttributeError 해결!
            t = threading.Thread(target=start_irq_loop, args=(new_loop,), daemon=True)
            t.start()
            print("🌐 [INFO] Shared IRQ loop started.")

        if hasattr(weight_tensor, 'int_repr'):
            w_np = weight_tensor.int_repr().detach().cpu().numpy()
            self.w_scale = weight_tensor.q_per_channel_scales()
            self.w_zero_point = weight_tensor.q_per_channel_zero_points()
        elif hasattr(weight_tensor, 'detach'):
            w_np = weight_tensor.detach().cpu().numpy()
            self.w_scale = 1.0  # 기본값
            self.w_zero_point = 0
        else:
            w_np = weight_tensor
            self.w_scale = 1.0  # 기본값
            self.w_zero_point = 0

        self.out_features = w_np.shape[0] # 예: 2304
        self.in_features = w_np.shape[1]  # 예: 768

        if hasattr(self.w_scale, 'detach'):
            # torch.Tensor인 경우
            w_scale_np = self.w_scale.detach().cpu().numpy().astype(np.float32)
        else:
            # float/numpy인 경우
            w_scale_np = np.asarray(self.w_scale, dtype=np.float32)

        if w_scale_np.ndim == 0:
            w_scale_np = np.full(self.out_features, float(w_scale_np), dtype=np.float32)
        x_scale_f   = float(x_scale)
        out_scale_f = float(self.out_scale)
        m_scale_per_channel = (x_scale_f * w_scale_np / out_scale_f).astype(np.float32)
        if bias_tensor is not None:
            if hasattr(bias_tensor, 'detach'):
                self.bias = (bias_tensor.detach().cpu().numpy().astype(np.float32)
                            / out_scale_f)
            else:
                self.bias = (np.asarray(bias_tensor, dtype=np.float32)
                            / out_scale_f)
        else:
            self.bias = np.zeros(self.out_features, dtype=np.float32)
        w_slices = np.vsplit(w_np, 4)
        m_slices = np.split(m_scale_per_channel, 4)
        b_slices = np.split(self.bias, 4)

        self.src2_list = []
        self.src2_c_list = []
        self.param_buf_list = []
        for w_s, m_s, b_s in zip(w_slices, m_slices, b_slices):
            # TPU 전용 전처리 (기존 함수)
            current_rows = w_s.shape[0]
            remainder = current_rows % 16
            if remainder != 0:
                padding_size = 16 - remainder
                # np.pad를 사용하여 아래쪽(axis=0)에 0을 추가
                # ( (위쪽_패딩, 아래쪽_패딩), (왼쪽_패딩, 오른쪽_패딩) )
                w_s = np.pad(w_s, ((0, padding_size), (0, 0)), mode='constant', constant_values=0)
                m_s = np.pad(m_s, (0, padding_size), mode='constant')
                b_s = np.pad(b_s, (0, padding_size), mode='constant')
                print(f"Padding added: {current_rows} -> {w_s.shape[0]}")

            w_s_tensor = torch.from_numpy(w_s).to(torch.int8)
            W, W_c = preprocess_weight_for_tpu(w_s_tensor)
            s2_c = allocate(shape=W_c.shape, dtype=np.int8)
            s2_c[:] = W_c
            self.src2_list.append(W.shape)
            self.src2_c_list.append(s2_c)

            num_ch = w_s.shape[0]
            interleaved = np.empty(num_ch * 2, dtype=np.float32)
            interleaved[0::2] = m_s
            interleaved[1::2] = b_s
            param_buf = allocate(shape=(num_ch * 2,), dtype=np.float32)
            param_buf[:] = interleaved
            param_buf.flush()
            self.param_buf_list.append(param_buf)
        # __init__에 추가
            self.result_buf = np.empty(
                (197*self.hw.batch_size, self.out_features), dtype=np.int8
            )


            self.result_torch = torch.from_numpy(self.result_buf)
            padded_rows = (197 * self.hw.batch_size + 7) // 8 * 8  # 400

            self.padded_input_map = {
                768:  np.zeros((padded_rows, 768),  dtype=np.int8),
                3072: np.zeros((padded_rows, 3072), dtype=np.int8),
            }

    def forward(self, x):
        t0 = time.perf_counter()
        original_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        num_rows = x_2d.shape[0] #197
        in_features = x_2d.shape[1] #768 or 3072
        padded_rows = (num_rows + 7) // 8 * 8
        flat_data = x_2d.int_repr().numpy().ravel()
        num_elements = flat_data.size
        t_a = time.perf_counter()
        import ctypes
        # 방법 1: ctypes memmove (가장 빠름)
        ctypes.memmove(
            self.hw.ip_buf_act.ctypes.data,
            flat_data.ctypes.data,
            flat_data.nbytes
        )
        t_b = time.perf_counter()

        # print(f"copy: {(t_b-t_a)*1000:.2f}ms")
        in_features = x.shape[-1]  # 768 or 3072
        padded_input = self.padded_input_map[in_features]
        current_input = padded_input
        # 인터럽트 감시 시작
        Interrupt_write(self.INTERRUPT1)
        irq_future = asyncio.run_coroutine_threadsafe(
            anext(interrupt_monitor(self.INTERRUPT1, num_events=4)),
            self.hw.irq_loop
        )
        t1 = time.perf_counter()
        for i in range(4):
            tpu_node = getattr(self.hw.ip_ol, f'TPU_PROCESSOR_{i}')
            #run_sa(tpu_node, current_input, self.hw.ip_buf_act.device_address, self.src2_list[i], self.src2_c_list[i], self.hw.ip_buf_dst[i],self.param_buf_list[i],x.q_zero_point(),self.out_zp)
            run_sa(tpu_node, current_input, self.hw.ip_buf_act.device_address, self.src2_list[i], self.src2_c_list[i], self.hw.ip_buf_dst[i],self.param_buf_list[i],x.q_zero_point(),self.out_zp)



        # 인터럽트 대기 (타임아웃 5초로 넉넉하게 설정)
        status = irq_future.result(timeout=5000)
        t2 = time.perf_counter()
        if(status==None):
            read_value = self.INTERRUPT1.read(0x00)
            print(f"read_value is {read_value}")
            breakpoint()
            raise RuntimeError(f"TPU HW Timeout! Interrupt status: {hex(read_value)}")

        actual_results_elements = padded_rows * (self.src2_list[0][1])
        if status is not None:
            # 안전하게 리스트 컴프리헨션으로 복사
            col_size = self.out_features//4
            i=0
            for d in self.hw.ip_buf_dst:
                arr = np.asarray(d).ravel()[:actual_results_elements].reshape(padded_rows, self.src2_list[0][1])
                self.result_buf[:num_rows, i*col_size:(i+1)*col_size] =   arr[:num_rows, :col_size]
                i=i+1

        # torch 변환도 from_numpy로 zero-copy
        res_torch = self.result_torch[:num_rows, :self.out_features]  .reshape(x.shape[:-1] + (self.out_features,)).to(x.device)
        out_int = res_torch.to(torch.uint8)  # numpy 거치지 말고 직접
        out_quant = torch._make_per_tensor_quantized_tensor(
            out_int,
            scale      = float(self.out_scale),
            zero_point = int(self.out_zp)
        )


        t3 = time.perf_counter()
        elapsed_collect1 = t1-t0
        elapsed_collect2 = t2-t0
        elapsed_collect3 = t3-t2


        #print(f"mlp_linear1: {elapsed_collect1*1000:.1f}ms ")
        #print(f"mlp_linear2: {elapsed_collect2*1000:.1f}ms ")
        #print(f"mlp_linear3: {elapsed_collect3*1000:.1f}ms ")

        #current_dst = self.dst_list[0][:padded_rows, :self.out_features//4]
        ################################# for test matrix #########################################################
        #sw_results = []
        #input_for_sw = current_input[:num_rows].astype(np.int32) - x.q_zero_point()
        #for i in range(4):
        #    sw_results.append(input_for_sw @ self.src2_list[i].astype(np.int32))
        #sw_res = np.concatenate(sw_results, axis=1)
        #bias_np = np.concatenate([ np.array(pb)[1::2] for pb in self.param_buf_list])
        #test_fp_np = sw_res.astype(np.float32) * (float (x.q_scale()) * self.w_scale.numpy() ) + bias_np*self.out_scale
        #test_fp = torch.from_numpy(test_fp_np).float()
        #test = torch.quantize_per_tensor(
        #    test_fp,
        #    self.out_scale,
        #    self.out_zp,
        #    torch.quint8
        #)

        #test2 = test.reshape(x.shape[:-1] + (self.out_features,))
        #test_int = test.int_repr().numpy()
        #wrong = np.where( np.abs(res_np-test_int) > 1)

        #if (len(wrong[0]) != 0):
        #    print("wrong detected")
        #    print(res_np[wrong])
        #    print(test_int[wrong])
        #    print(len(wrong[0]))

        #    breakpoint()
        #else:
        #    print("✅ correct")

        return out_quant

'''
for i in range(48):  np.where(  self.src2_list[1][:,16*i:16*(i+1)] != self.src2_c_list[1][768*i:768*(i+1)] )


np.where(results[0] != sw_results[0])
np.where(results[1] != sw_results[1])
np.where(results[2] != sw_results[2])
np.where(results[3] != sw_results[3])

idx = 0
self.src2_c_list[idx].flush()

irq_future = asyncio.run_coroutine_threadsafe( anext(interrupt_monitor(self.INTERRUPT1, num_events=4)),  self.hw.irq_loop  )
TPU_info=run_sa(self.hw.ip_ol.TPU_PROCESSOR_0, current_input, self.input_buf.device_address,self.src2_list[idx], self.src2_c_list[idx], self.dst_list[idx])
status = irq_future.result(timeout=5000)
actual_results_elements = padded_rows * (self.out_features // 4)
self.dst_list[idx].invalidate()
temp_view = self.dst_list[idx].flat[:actual_results_elements]
temp_reshaped = temp_view.reshape(padded_rows, self.out_features // 4)
wrong1 = np.where(temp_reshaped[:num_rows] != results[idx])
wrong1
value1 = self.input_buf.copy()
self.input_buf.invalidate()
np.where(value1 != self.input_buf)

value1 = self.src2_list[idx].copy()
self.src2_list[idx].invalidate()
np.where(value1 != self.src2_list[idx])
save_int8_matrix_to_hex(self.src2_c_list[idx],"B_CONCAT.hex",16)
save_int8_matrix_to_hex(self.input_buf,"A.hex",16)
save_int32_array_to_hex(results[idx],"out.hex",4)

'''

