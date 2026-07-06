import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.nn.quantized import FloatFunctional
import numpy as np
import copy
import struct
from pynq import MMIO, allocate
from config.config import SCALE, HIDDEN_DIM, SEQ_LEN, BATCH_SIZE
import os, mmap, threading, time

# ==============================================================================
#  [Import Utils]
# ------------------------------------------------------------------------------
from collections import namedtuple

HIDDEN_DIM  = 768
SEQ_LEN     = 197
SEQ_LEN_PAD = 208
PACK        = 16
NPARTS      = HIDDEN_DIM // PACK

class BRAMBuffer:
    def __init__(self, phys_addr, mmio):
        self.device_address = phys_addr
        self._mmio = mmio    
        self._arr = np.frombuffer(mmio.array, dtype=np.uint8)
    
    def __array__(self):
        return self._arr
    
    def __getitem__(self, idx):
        return self._arr[idx]
    
    def __setitem__(self, idx, val):
        self._arr[idx] = val

    def reshape(self, *args, **kwargs):
        return self._arr.reshape(*args, **kwargs)
    
    def flatten(self, *args, **kwargs):
        return self._arr.flatten(*args, **kwargs)

class BufferManager:
    ACP_START    = 0x58000000
    ACP_END      = 0x5A000000
    WEIGHT_START = 0x5A000000

    def __init__(self):
        self._pre_acp_dummies  = []
        self._post_acp_dummies = []

    def reserve_pre_acp(self):
        """CMA를 0x38000000 직전까지 채움"""
        print("Pre-ACP 더미 채우는 중...")
        while True:
            buf = allocate(shape=(1024*4), dtype=np.uint8)
            if buf.device_address >= self.ACP_START:
                break
            self._pre_acp_dummies.append(buf)
        print(f"✅ Pre-ACP 완료: {len(self._pre_acp_dummies)}MB 점유")

    def reserve_post_acp(self):
        """ACP 이후를 0x40000000까지 채움"""
        print("Post-ACP 더미 채우는 중...")
        while True:
            buf = allocate(shape=(1024*4), dtype=np.uint8)
            if buf.device_address >= self.WEIGHT_START:
                break
            self._post_acp_dummies.append(buf)
        print(f"✅ Post-ACP 완료: {len(self._post_acp_dummies)}MB 점유")

    def free_dummies(self):
        """더미 해제 (모든 중요 버퍼 할당 후)"""
        for b in self._pre_acp_dummies + self._post_acp_dummies:
            b.freebuffer()
        self._pre_acp_dummies.clear()
        self._post_acp_dummies.clear()


# ==============================================================================
#  [FPGA Hardware Class]
# ------------------------------------------------------------------------------
class FPGAManager:
    def __init__(self, ip_path=None):
        print("[FPGA Manager] Initializing Hardware...")
        if ip_path is None:
            self.ip_ol = None
            self.ip_buf_inp = None
            self.ip_buf_gam = None
            self.ip_buf_bet = None
            self.ip_buf_out = None

            print("[FPGA Manager] IP Path is None.")
        else:
            row_nums = 208
            self.batch_size = BATCH_SIZE
            self.num_heads =12
            self.d_k = 64
            self.ip_ol = Overlay(ip_path)
            self.buf_mgr = BufferManager()
            self.buf_mgr.reserve_pre_acp()

            #####################ACP REGION ###################################################
            import math
            from collections import namedtuple
            # 개선: 주소만 담은 간단한 객체
            PhysAddr = namedtuple('PhysAddr', ['device_address'])

            #self.ip_buf_act = allocate(shape=(row_nums*BATCH_SIZE*3072), dtype=np.uint8,cacheable = True)
            self.ip_buf_dst = [allocate(shape=(row_nums*BATCH_SIZE, 1024), dtype=np.uint8 ,cacheable = True) for _ in range(4)]
            self.ip_qbuf_dst = [allocate(shape=(row_nums*BATCH_SIZE, 1024), dtype=np.uint8 ,cacheable = True) for _ in range(4)]
            self.ip_kbuf_dst = [allocate(shape=(row_nums*BATCH_SIZE, 1024), dtype=np.uint8 ,cacheable = True) for _ in range(4)]
            self.ip_vbuf_dst = [allocate(shape=(row_nums*BATCH_SIZE, 1024), dtype=np.uint8 ,cacheable = True) for _ in range(4)]
            #SOFTMAX_PINGPONG
            '''
            self.ip_buf_mm_OCM_list = [
                allocate(shape=(row_nums*row_nums), dtype=np.int8, cacheable = True)
                for _ in range(4*2)
            ]
            '''
        
            BRAM_BASE_ADDR = 0xB000_0000  # 예시
            BRAM_SIZE      = row_nums * row_nums  # bytes (int8)
            
            # 각 버퍼를 BRAM 고정 오프셋에 매핑
            self.ip_buf_mm_OCM_list = []
            for i in range(16):
                offset = i * row_nums * row_nums
                phys_addr = BRAM_BASE_ADDR + offset
                mmio = MMIO(phys_addr, BRAM_SIZE)
                self.ip_buf_mm_OCM_list.append(BRAMBuffer(phys_addr, mmio))
            
            self.ocm_np = [buf.reshape(row_nums, row_nums)
                  for buf in self.ip_buf_mm_OCM_list[:16]]

            # torch도 각각 from_numpy로 view 유지
            self.ocm_u8_torch = [torch.from_numpy(v) for v in self.ocm_np]
            

            self.slot   = (row_nums // 16) * 64  * 16  # 13 * 64 * 16 = 13312
            self.slots  = BATCH_SIZE * 12             # 24
            total  = self.slots * self.slot                   # 24 * 13312 = 319488
            # 4KB 정렬 slot_size
            self.slot_aligned = math.ceil(self.slot / 4096) * 4096  # 16384


            # ─ 3) ACP 이후 더미로 채우기 ──────────────────────────
            self.buf_mgr.reserve_post_acp()
            
            #########DDR REGION
            self.ip_buf_act = allocate(shape=(row_nums*BATCH_SIZE*3072), dtype=np.uint8,cacheable = False)
            #self.ip_buf_dst2 = [allocate(shape=(row_nums*BATCH_SIZE, 1024), dtype=np.uint8 ,cacheable = False) for _ in range(4)]
            #self.ip_buf_mm_Q_list = [
            #    allocate(shape=(row_nums*64), dtype=np.uint8, cacheable = False)  # head당 [197, 64]
            #    for _ in range(12*BATCH_SIZE)
            #]

            self.ip_buf_mm_KT_all = allocate(
                shape=(self.slots * self.slot_aligned,), dtype=np.int8
            )
            # KT scratch: [2, 12, 13, 64, 16] contiguous
            self._KT_scratch = np.empty(
                (BATCH_SIZE, 12, row_nums//16, 64, 16),
                dtype=np.int8
            )

            self.ip_buf_mm_KT_list = [
                PhysAddr(device_address=
                    self.ip_buf_mm_KT_all.device_address + idx * self.slot_aligned)
                for idx in range(self.slots)
            ]

            self.kt_all_np = np.frombuffer(
                self.ip_buf_mm_KT_all, dtype=np.int8
            )
            self.kt_strided = np.lib.stride_tricks.as_strided(
                self.kt_all_np,
                shape=(self.slots, self.slot),           # [24, 13312]
                strides=(self.slot_aligned, 1)      # head간 16384 bytes 점프
            )
            ########################################    QQQ            #######################################
            self.slot_q         = row_nums * 64          # 208 * 64 = 13312
            self.slot_q_aligned = math.ceil(self.slot_q / 4096) * 4096  # 16384

            self.ip_buf_mm_Q_all = allocate(
                shape=(BATCH_SIZE * 12 * self.slot_q_aligned,),
                dtype=np.uint8
            )
            self.ip_buf_mm_Q_list = [
                PhysAddr(device_address=
                    self.ip_buf_mm_Q_all.device_address + idx * self.slot_q_aligned)
                for idx in range(BATCH_SIZE * 12)
            ]

            self._Q_slot         = self.slot_q
            self._Q_slot_aligned = self.slot_q_aligned
            self._Q_slots        = BATCH_SIZE * self.num_heads

            q_all_np = np.asarray(self.ip_buf_mm_Q_all)
            self.q_strided = np.lib.stride_tricks.as_strided(
                q_all_np,
                shape=(self._Q_slots, row_nums, self.d_k),  # (24, 208, 64)
                strides=(self.slot_q_aligned, self.d_k, 1)
            )
            
            ######################################     VVVVVV  ####################################
            self.slot_v         = row_nums * 64          # 208 * 64 = 13312
            self.slot_v_aligned = math.ceil(self.slot_v / 4096) * 4096  # 16384

            self.ip_buf_mm_V_all = allocate(
                shape=(BATCH_SIZE * self.num_heads * self.slot_v_aligned,),
                dtype=np.int8
            )
            self._V_scratch = np.empty(
                (BATCH_SIZE, self.num_heads, row_nums, 64//16, 16),
                dtype=np.int8
            )
            self.ip_buf_mm_V_list = [
                PhysAddr(device_address=
                    self.ip_buf_mm_V_all.device_address + idx * self.slot_v_aligned)
                for idx in range(BATCH_SIZE * self.num_heads)
            ]
            self._V_slot         = self.slot_v
            self._V_slot_aligned = self.slot_v_aligned
            self._V_slots        = BATCH_SIZE * self.num_heads

            self.v_all_np  = np.asarray(self.ip_buf_mm_V_all).view(np.int8)
            self.v_strided = np.lib.stride_tricks.as_strided(
                self.v_all_np,
                shape=(self.batch_size, self.num_heads, 64//16, row_nums, 16),
                strides=(
                    self.num_heads * self._V_slot_aligned,
                    self._V_slot_aligned,
                    row_nums*16,
                    16,
                    1
                )
            )

            self.slot_P  = row_nums * row_nums
            self.slots_P = BATCH_SIZE * 12

            self.ip_buf_mm_P_list = [
                PhysAddr(...) for i in range(self.slots_P)  # 24개
            ]
            self.ip_buf_mm_P_all = allocate(
                shape=(BATCH_SIZE, 12, row_nums, row_nums),
                dtype=np.int8
            )
            self.ip_buf_mm_P_list = [
                PhysAddr(device_address=
                    self.ip_buf_mm_P_all.device_address + i * self.slot_P
                )
                for i in range(self.slots_P)  # ← 여기서 slots_P가 제대로 설정됐는지 확인
            ]
            P_all_torch = torch.from_numpy(np.asarray(self.ip_buf_mm_P_list))
            self.P_strided = torch.from_numpy(  np.asarray(self.ip_buf_mm_P_all)).view(torch.qint8) 
        

            self.pv_result_memory = np.empty(
                (BATCH_SIZE * 12, 208, 64), dtype=np.uint8)
            self._pv_result_torch = torch.from_numpy(self.pv_result_memory)

            self._pv_result_view = self._pv_result_torch.reshape(BATCH_SIZE, 12, 208, 64)

 

            # Per-thread scratch buffers (사전할당, 평생 재사용)

            self._softmax_scratch_f32 = np.empty((4, 208, 208), dtype=np.float32)
            self._softmax_scratch_u8  = np.empty((4, 208, 208), dtype=np.uint8)
            self._softmax_scratch_f32_torch = torch.from_numpy(self._softmax_scratch_f32)
            self._softmax_scratch_u8_torch  = torch.from_numpy(self._softmax_scratch_u8)
            
            self._ort_np_buf = np.zeros(
                (4,) + self._softmax_scratch_f32_torch.shape[1:],
                dtype=np.float32
            )

            batch = BATCH_SIZE

            self.data_a_buf = allocate(shape=(batch*SEQ_LEN_PAD*HIDDEN_DIM,),
                                       dtype=np.uint8, cacheable=False)
            self.data_b_buf = allocate(shape=(batch*SEQ_LEN_PAD*HIDDEN_DIM,),
                                       dtype=np.uint8, cacheable=False)
            self.data_c_buf = allocate(shape=(batch*SEQ_LEN_PAD*HIDDEN_DIM,),
                                       dtype=np.uint8, cacheable=False)
            self.param_buf  = allocate(shape=(HIDDEN_DIM*2,),
                                       dtype=np.float32, cacheable=False)
            
            self.data_a_np = np.asarray(self.data_a_buf).reshape(
                batch, NPARTS, SEQ_LEN_PAD, PACK)
            self.data_b_np = np.asarray(self.data_b_buf).reshape(
                batch, NPARTS, SEQ_LEN_PAD, PACK)
            self.data_c_np = np.asarray(self.data_c_buf).reshape(
                batch, NPARTS, SEQ_LEN_PAD, PACK)

            self.data_a__view = self.data_a_buf.transpose(0, 2, 1, 3)
            self.data_b__view = self.data_b_buf.transpose(0, 2, 1, 3)
            self.data_c__view = self.data_c_buf.transpose(0, 2, 1, 3)

            self.param_buf_np = np.asarray(self.param_buf)

            self.ln_result_buf   = np.empty((batch, SEQ_LEN, HIDDEN_DIM), dtype=np.uint8)
            self.ln_result_torch = torch.from_numpy(self.ln_result_buf)

            ln_ip = self.ip_ol.layernorm_0
            ln_ip.write()
            
            ln_ip.register_map.inp_a  = self.ip_buf_inp.device_address
            ln_ip.register_map.inp_a  = self.ip_buf_inp.device_address
            ln_ip.register_map.inp_a  = self.ip_buf_inp.device_address
            ln_ip.register_map.par_0  = self.ip_buf_par.device_address
            ln_ip.register_map.out_0  = self.ip_buf_out.device_address
            ln_ip.register_map.batch  = batch
            ln_ip.register_map.seqlen = SEQ_LEN
            ln_ip.register_map.dim    = HIDDEN_DIM

            print("[FPGA Manager] Initialization Complete")
    
    def free(self):
        """free allocated buffers"""
        pass


    #      ###  #   # ##### ####  #   #  ###  ####  #   # 
    #     #   #  # #  #     #   # ##  # #   # #   # ## ## 
    #     #####   #   ####  ####  # # # #   # ####  # # # 
    #     #   #   #   #     #  #  #  ## #   # #  #  #   # 
    ##### #   #   #   ##### #   # #   #  ###  #   # #   # 

def float_packint(value):
    return struct.unpack('<I', struct.pack('<f', float(value)))[0]

def run_layernorm_hw(
    hw,
    x,
    eps,
    scale_a, zp_a,
    scale_b, zp_b,
    scale_c, zp_c,
    scale_o, zp_o,
):
    orig_dtype = x.dtype
    B, N, C    = x.shape

    in_scale = float(x.q_scale())
    in_zp    = int(x.q_zero_point())

    ln_ip = hw.ip_ol.layernorm_0
    q_inp = x.int_repr().numpy()

    np.copyto(
        hw.inp_data_view[:, :N, :, :],
        q_inp.reshape(B, N, NPARTS, PACK)
    )

    ln_ip.register_map.CTRL.AP_START = 1
    while not ln_ip.register_map.CTRL.AP_DONE:
        pass

    np.copyto(
        hw.ln_result_buf.reshape(B, N, NPARTS, PACK),
        hw.out_data_view[:, :N, :, :]
    )

    return torch._make_per_tensor_quantized_tensor(
        hw.ln_result_torch,
        float(out_scale),
        int(out_zp))


class fusedResidualLayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape,
        eps,
        hw,
        scale_a, zp_a,
        scale_b, zp_b,
        scale_c, zp_c,
        scale_o, zp_o,
    ):
        self.normalized_shape = normalized_shape
        self.eps              = eps
        self.hw               = hw
        
        # add input
        self.scale_a, self.zp_a = scale_a, zp_a
        self.scale_b, self.zp_b = scale_b, zp_b
        
        # add output === layernorm input
        self.scale_c, self.zp_c = scale_c, zp_c
        
        # layernorm output
        self.scale_o, self.zp_o = scale_o, zp_o
        
        # layernorm parameters
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias   = nn.Parameter(torch.zeros(normalized_shape))
        
        # layernorm parameters numpy
        self._weight_np = None
        self._bias_np   = None

    def sync_params(self):
        self._weight_np = self.weight.detach().cpu().numpy().astype(np.float32)
        self._bias_np   = self.bias.detach().cpu().numpy().astype(np.float32)

    def forward(self, x):
        if self._weight_np is None:
            self.sync_params()

        C = self.normalized_shape[0]
        np.copyto(self.hw.par_buf_np[:C], self._weight_np)
        np.copyto(self.hw.par_buf_np[C:], self._bias_np)

        out_tensor = run_layernorm_hw(
            self.hw,
            x,
            self.eps,
            self.out_scale,
            self.out_zp,
        )

        return out_tensor


class QuantLayerNormFPGA(nn.Module):
    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-6,
        hw=None,
        out_scale: 'float | None' = None,
        out_zp: 'int | None' = None
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps              = eps
        self.hw               = hw
        self.out_scale        = out_scale
        self.out_zp           = out_zp

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias   = nn.Parameter(torch.zeros(normalized_shape))

        self._weight_np: np.ndarray | None = None
        self._bias_np:   np.ndarray | None = None

    def sync_params(self):
        self._weight_np = self.weight.detach().cpu().numpy().astype(np.float32)
        self._bias_np   = self.bias.detach().cpu().numpy().astype(np.float32)

    def forward(self, x):
        if self._weight_np is None:
            self.sync_params()

        C = self.normalized_shape[0]
        np.copyto(self.hw.par_buf_np[:C], self._weight_np)
        np.copyto(self.hw.par_buf_np[C:], self._bias_np)

        out_tensor = run_layernorm_hw(
            self.hw,
            x,
            self.eps,
            self.out_scale,
            self.out_zp,
        )

        return out_tensor


# ==============================================================================
#  [Custom Quantized Multihead Attention Class]
# ------------------------------------------------------------------------------
class QuantMultiheadAttention(nn.Module):
    def __init__(self, dim, num_heads, attn_drop=0.0, proj_drop=0.0, use_hw=False):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim       = dim
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.use_hw    = use_hw

        # Layers
        self.qkv       = nn.Linear(dim, 3 * dim, bias=True)
        self.proj      = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.func = FloatFunctional()

        # Buffers for Sparsity Masks (persistent=False: state_dict에 저장 안 함)
        self.qkv.register_buffer("_mixed_sparsity_mask", None, persistent=False)
        self.proj.register_buffer("_mixed_sparsity_mask", None, persistent=False)
        
        # HW Initialization (필요시)
        # if self.use_hw:
        #     FPGAManager().initialize() # Ensure `HW is ready

    def _enforce_weight_mask(self):
        # QKV Mask
        qkv_mask = getattr(self.qkv, "qkv_mask", None)
        if qkv_mask is not None:
            self.qkv.weight.data.mul_(qkv_mask)
            
        # Proj Mask
        proj_mask = getattr(self.proj, "proj_mask", None)
        if proj_mask is not None:
            self.proj.weight.data.mul_(proj_mask)

    def forward(self, query, key, value, need_weights=False, attn_mask=None, **kwargs):
        # Apply mask
        self._enforce_weight_mask()
        x = query
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        # QKV Projection and Reshape
        qkv = self.qkv(x)  # [B, N, 3*C]
        if hasattr(x, 'node'): 
            print("Currently in Tracing Mode (Proxy)")
        else:
            print(f"Actual Data Flow - Type: {x.dtype}")
            print(f"Actual Data Flow - Type: {qkv.dtype}")
            if x.is_quantized:
                print(f"Quantized: {x.qscheme()}, Scale: {x.q_scale()}")
                print(f"Quantized: {qkv.qscheme()}, Scale: {qkv.q_scale()}")
                
        qkv = qkv.reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, N, D]

        # Attention Score
        attn = self.func.matmul(q, k.transpose(-2, -1))
        attn = self.func.mul(attn, self.scale)

        if attn_mask is not None:
            attn = self.func.add(attn, attn_mask)
        # Softmax
        if self.use_hw:
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)

        # Output Projection
        out = self.func.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().reshape(B, N, C)  # [B,N,C]
        out = self.proj(out)
        out = self.proj_drop(out)

        return out, None


def replace_ln_to_fpga(model, hw):
    for node in model.graph.nodes:
        is_ln_module = (node.op == "call_module" and isinstance(model.get_submodule(node.target), nn.LayerNorm))
        is_ln_function = (node.op == "call_function" and node.target == torch.nn.functional.layer_norm)

        if is_ln_module or is_ln_function:
            if is_ln_module:
                orig_ln = model.get_submodule(node.target)
                weight = orig_ln.weight.data
                bias = orig_ln.bias.data
                eps = orig_ln.eps
                normalized_shape = orig_ln.normalized_shape
                ln_input_node = node.args[0]
            else:
                ln_input_node = node.args[0]
                normalized_shape = node.args[1]
                weight = model.get_parameter(node.args[2].target).data if node.args[2].op == "get_attr" else node.args[2]
                bias = model.get_parameter(node.args[3].target).data if node.args[3].op == "get_attr" else node.args[3]
                eps = node.kwargs.get("eps", 1e-5) if "eps" in node.kwargs else (node.args[4] if len(node.args) > 4 else 1e-5)

            out_scale = model.get_submodule(node.target).scale
            out_zp = model.get_submodule(node.target).zero_point
            
            fpga_ln = QuantLayerNormFPGA(
                normalized_shape=normalized_shape,
                eps=eps,
                hw=hw,
                out_scale=out_scale,
                out_zp=out_zp
            )
            fpga_ln.weight.data.copy_(weight)
            fpga_ln.bias.data.copy_(bias)

            new_module_name = f"fpga_layernorm_{node.name}"
            model.add_module(new_module_name, fpga_ln)

            with model.graph.inserting_after(node):
                new_node = model.graph.call_module(new_module_name, args=(ln_input_node,))
                node.replace_all_uses_with(new_node)

            model.graph.erase_node(node)

    model.graph.lint()
    model.recompile()
    return model


# ==============================================================================
#  Helper Functions
# ------------------------------------------------------------------------------
def replace_layernorm_to_fpga(module: nn.Module, hw: FPGAManager, use_hw=False):
    replaced_count = 0
    
    for name, child in module.named_children():
        replaced_count += replace_layernorm_to_fpga(child, hw, use_hw)
        
        if isinstance(child, nn.LayerNorm):
            new_ln = QuantLayerNormFPGA(
                normalized_shape=child.normalized_shape,
                eps=child.eps,
                use_hw=use_hw,
                hw=hw
            )
            
            if child.weight is not None:
                new_ln.weight.data = child.weight.data.clone()
            if child.bias is not None:
                new_ln.bias.data = child.bias.data.clone()
            
            # 3. 모듈 교체
            setattr(module, name, new_ln)
            replaced_count += 1
            
    return replaced_count
