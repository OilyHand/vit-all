#!/usr/bin/env python3
import numpy as np
import math
import struct, time
import os, struct, select, threading, torch ,asyncio
from pynq import allocate 

def Interrupt_write(INTERRUPT):

    IER_OFFSET = 0x08
    MER_OFFSET = 0x1c
    INTERRUPT.write(IER_OFFSET,0x1)
    INTERRUPT.write(MER_OFFSET,0x3)
    read_val1 = INTERRUPT.read(IER_OFFSET)
    read_val2 = INTERRUPT.read(MER_OFFSET)

    if read_val1==0x1 and read_val2 == 0x3:
        return 1
    else:
        return 0


async def interrupt_monitor(INTERRUPT, num_events, timeout_ms=5000, ocm_regions=None):
    """
    ocm_regions: [(ptr, size), ...] 각 TPU의 OCM 버퍼 주소/크기
    신호 포착 시 해당 OCM cache invalidate
    """
    ISR_OFFSET = 0x00
    IAR_OFFSET = 0x0C
    check_interval = 0.0001
    start_wait = time.perf_counter()
    acc_reg    = 0
    prev_acc   = 0
    target_flag = (1 << num_events) - 1

    while True:
        reg_val  = INTERRUPT.read(ISR_OFFSET)
        acc_reg |= reg_val

        # 새로 포착된 비트만 확인
        new_bits = acc_reg & ~prev_acc
        if new_bits and ocm_regions:
            for i in range(num_events):
                if new_bits & (1 << i):
                    ptr, size = ocm_regions[i]
                    _invalidate_cache(ptr, size)   # ← 해당 OCM만 invalidate

        prev_acc = acc_reg

        if (acc_reg & target_flag) == target_flag:
            INTERRUPT.write(IAR_OFFSET, acc_reg)
            yield acc_reg
            return

        if (time.perf_counter() - start_wait) > (timeout_ms / 1000):
            print(f"[IRQ] Polling Timeout {bin(acc_reg)}")
            yield None
            return

        await asyncio.sleep(check_interval)



# ARM64 전용 cache invalidate
def _invalidate_cache(ptr, size):
    # __aarch64_sync_cache_range 대신
    # PYNQ의 libxlnk_cma.so 사용
    import ctypes
    
    # 대신 가장 근접한 방법:
    libc = ctypes.CDLL("libc.so.6")
    # msync로 동기화
    libc.msync(ctypes.c_void_p(ptr), 
               ctypes.c_size_t(size), 
               ctypes.c_int(4))  # MS_INVALIDATE = 4

def prepare_weight_blocks(
    B,
    block_cols=16,
    base_offset=0x0000_0000,
    base_stride=0x0010_0000
):
    """
    B: (K, N) uint8 matrix (allocate buffer여도 되고 numpy여도 됨)
    block_cols: column block 크기
    base_offset: 시작 주소
    base_stride: 블록 간 간격
    """

    K, N = B.shape
    num_blocks = N // block_cols

    total_size = base_offset + base_stride * num_blocks
    weight_ddr = allocate(shape=(total_size,), dtype=np.uint8)

    packed_blocks = []

    for i in range(num_blocks):

        # 1️⃣ column block 분할
        B_block = B[:, i*block_cols:(i+1)*block_cols]

        # 2️⃣ delta encoding
        addr_delta, values = to_delta_stream(B_block)

        # 3️⃣ packing
        packed = pack_delta_8_8(addr_delta, values, B_block.shape, True)

        packed_blocks.append(packed)

        # 4️⃣ offset 계산
        offset = base_offset + i * base_stride

        # 5️⃣ 복사
        copy_to_offset(weight_ddr, offset, packed)

    return weight_ddr, packed_blocks

def copy_to_offset(buf, offset, packed):
    size = packed.nbytes
    buf[offset:offset+size] = packed.view(np.uint8)



def w32(mmio, offset: int, value: int):
    mmio.write(offset, int(value) & 0xFFFFFFFF)

def r32(mmio, offset: int) -> int:
    return int(mmio.read(offset))

def pack_cont1(relu: int, weight_width: int, weight_height: int) -> int:
    relu = 1 if relu else 0
    return ((relu & 1) << 31) | ((weight_width & 0x7FFF) << 16) | (weight_height & 0xFFFF)

def pack_cont2(relu: int, act_height: int) -> int:
    relu = 1 if relu else 0
    return ((relu & 1) << 31) | (act_height & 0xFFFF)

def _get_phys(buf) -> int:
    # PYNQ allocate 버퍼는 .physical_address 제공
    if not hasattr(buf, "physical_address"):
        raise TypeError("Buffer must be allocated by pynq.allocate() (needs .physical_address).")
    return int(buf.physical_address)

def _shape2(buf):
    if not hasattr(buf, "shape") or len(buf.shape) != 2:
        raise ValueError("Matrix must be a 2D array (shape=(rows, cols)).")
    return int(buf.shape[0]), int(buf.shape[1])




def save_matrix_to_bin_chunked(matrix, filename):
    """
    int8 matrix를 그대로 BIN으로 저장 (엔디안/리오더링 없음)

    matrix: numpy array (1D or 2D) 또는 array-like
    filename: 출력 파일 경로 (예: "A.bin")
    """
    arr = np.asarray(matrix, dtype=np.uint8).ravel()  # row-major로 1D flatten
    print(arr)
    with open(filename, "wb") as f:
        f.write(arr.tobytes())

def save_matrix_to_bin_chunked2(matrix, filename):
    """
    uint16 배열을 uint8 두 개씩으로 나눠 BIN으로 저장
    (little-endian 기준)
    """
    arr16 = np.asarray(matrix, dtype=np.uint16).ravel()

    # uint8로 reinterpret (각 uint16 → uint8 2개)
    arr8 = arr16.view(np.uint8)

    with open(filename, "wb") as f:
        f.write(arr8.tobytes())

def calculate_sparsity(arr):
    arr = np.asarray(arr)
    total_elements = arr.size
    zero_elements = np.sum(arr == 0)
    sparsity_percent = (zero_elements / total_elements) * 100
    return sparsity_percent

def save_int32_array_to_hex(matrix, filename, chunk=8):
    """
    (N,) 또는 (N x M) int32 배열을 HEX 텍스트 파일로 저장.
    한 줄에 chunk개씩, 각 값을 8자리 HEX로 기록 (32bit).
    """
    arr = np.asarray(matrix, dtype=np.int32).ravel()

    with open(filename, "w") as f:
        for i in range(0, len(arr), chunk):
            block = arr[i:i + chunk]
            hex_line = ""
            for val in reversed(block):
                # 2's complement 보존을 위해 uint32로 캐스팅
                hex_line += format(np.uint32(val), "08X")
            f.write(hex_line + "\n")

def save_uint16_array_to_hex_chunked(matrix, filename, chunk=16):
    """
    (N,) 또는 (N x M) uint16 배열을 HEX 텍스트 파일로 저장.
    한 줄에 chunk개씩, 각 값을 4자리 HEX로 기록.
    """
    arr = np.asarray(matrix, dtype=np.uint16).ravel()  # 1D로 평탄화

    with open(filename, "w") as f:
        hex_line = ""
        block = arr[0:2]
        for val in reversed(block):
            hex_line += format(int(val), "04X")  # 16비트 기준 4자리 HEX
        f.write(hex_line + "\n")

        hex_line = ""
        block = arr[2:4]
        for val in reversed(block):
            hex_line += format(int(val), "04X")  # 16비트 기준 4자리 HEX
        f.write(hex_line + "\n")

        for i in range(4, len(arr), chunk):
            block = arr[i:i+chunk]
            hex_line = ""
            # 필요한 경우 순서 뒤집어서 저장
            for val in reversed(block):
                hex_line += format(int(val), "04X")  # 16비트 기준 4자리 HEX
            f.write(hex_line + "\n")

def save_int8_matrix_to_hex_chunked(matrix, filename, chunk=16):
    """
    (N x N) uint8/Int8 행렬을 HEX 텍스트 파일로 저장.
    한 줄에 chunk개씩, 각 값을 2자리 HEX로 기록.
    """
    matrix = np.asarray(matrix)
    with open(filename, "w") as f:
        for row in matrix:
            for i in range(0, len(row), chunk):
                block = row[i:i+chunk]
                hex_line = ""
                # 필요한 경우 순서 뒤집어서 저장
                for val in reversed(block):
                    u8 = np.uint8(val)
                    hex_line += format(u8, "02X")
                f.write(hex_line + "\n")


def generate_sparse_uint8_matrix(size1=1000,size2=1000, sparsity=0.5):
    """
    size: 행렬 크기 (size x size)
    sparsity: 0의 비율 (예: 0.5 → 50% zero)
    데이터 타입: uint8 (0~255)
      - 0은 '빈 값', 1~255는 non-zero로 사용
    """
    total = size1 * size2
    num_zero = int(total * sparsity)
    num_nonzero = total - num_zero

    mat = allocate(shape = (size1,size2), dtype = np.uint8)


    # zero + non-zero 결합
    data = np.concatenate([
        np.zeros(num_zero, dtype=np.uint8),
        np.random.randint(1, 256, size=num_nonzero, dtype=np.uint8)
    ])

    # 무작위 셔플 후 행렬로 reshape
    np.random.shuffle(data)
    mat[:] = data.reshape(size1, size2)
    return mat

def generate_sparse_int8_matrix(size1=1000,size2=1000, sparsity=0.5):
    """
    size: 행렬 크기 (size x size)
    sparsity: 0의 비율 (예: 0.5 → 50% zero)
    데이터 타입: uint8 (0~255)
      - 0은 '빈 값', 1~255는 non-zero로 사용
    """
    total = size1 * size2
    num_zero = int(total * sparsity)
    num_nonzero = total - num_zero

    # non-zero 값을 1~255로 생성 (0은 진짜 '0'으로만 사용)
    mat = allocate(shape = (size1,size2), dtype = np.int8)
    nonzero_values = np.random.randint(-128,128 ,size=num_nonzero, dtype=np.int8)

    # zero + non-zero 결합
    data = np.concatenate([
        np.zeros(num_zero, dtype=np.int8),
        nonzero_values
    ])

    # 무작위 셔플 후 행렬로 reshape
    np.random.shuffle(data)
    mat[:] = data.reshape(size1, size2)
    return mat


def to_delta_stream(B):
    """
    B: 2D numpy array (uint8 / int8)

    1D(row-major)로 평탄화한 뒤,
      - 0이 아닌 값들 사이의 '0 개수'를 delta로 사용해서
      - (delta(0~15), value(0~255)) 스트림으로 변환.

    인코딩 규칙:
      - flat을 왼쪽부터 끝까지 스캔.
      - zero_run: 직전 이벤트 이후로 나온 0 개수.
      - v != 0을 만나면:
          while zero_run > 15:
              (delta=15, value=0) 출력  → 긴 0 구간 잘라내기
              zero_run -= 15
          (delta=zero_run, value=v) 출력
          zero_run = 0
      - 끝까지 갔을 때 남은 zero_run은
        굳이 토큰으로 안 내보내도 됨(맨 끝 0들은 그냥 0으로 남겨두면 됨).

    디코딩 규칙:
      pos = -1
      flat = zeros(total_size)
      for (delta, value) in stream:
          pos += delta
          if value != 0:
              pos += 1
              flat[pos] = value

    return:
      - addr_delta: uint8 배열, 각 원소는 0~15 (4bit 사용 가능)
      - values    : uint8 배열, 같은 길이, 0은 '긴 zero run 연장 토큰'으로 사용
    """
    B = np.asarray(B)
    flat = B.ravel().astype(np.uint8)
    total_size = flat.size

    deltas = []
    vals   = []

    zero_run = 0

    for v in flat:
        if v == 0:
            # 0이면 run 길이만 늘림
            zero_run += 1
            if zero_run > 15:
                deltas.append(15)   # 15개의 0
                vals.append(0)      # 이 토큰은 "0만 있음" 의미
                zero_run = 0
        else:
            # 남은 0 개수 (0~15)에 대해 non-zero 토큰 하나 출력
            deltas.append(zero_run)
            vals.append(int(v))

            zero_run = 0


    addr_delta = np.array(deltas, dtype=np.uint8)
    values     = np.array(vals,   dtype=np.uint8)
    return addr_delta, values






import numpy as np

def pack_delta_8_8(addr_delta, values, shape, add_header=False):
    """
    delta 포맷:
      [15:8] = delta(8bit, uint8)
      [ 7:0] = value(8bit, uint8)

    addr_delta: int32/uint8 배열 (delta)
    values    : uint8 배열 (non-zero 값)
    -> 1D uint16 배열 반환

    add_header:
      - False: 토큰들만 16bit 배열로 반환
      - True : 앞에 64bit(= 네 개의 uint16) 헤더를 붙여서 반환
               헤더 구조 (uint16 4개, 내부 배열 순서):
                 [0] payload_bytes_low   (payload_bytes[15:0])
                 [1] payload_bytes_high  (payload_bytes[31:16])
                 [2] rows                (원래 B.shape[0], 16bit)
                 [3] cols                (원래 B.shape[1], 16bit)
               이후에 token word16들이 이어짐.
    """
    addr_delta = np.asarray(addr_delta)
    values     = np.asarray(values)
    assert addr_delta.shape == values.shape

    if np.any(addr_delta < 0) or np.any(addr_delta > 255):
        print("[WARN] some delta > 255; 포맷 확장이 필요할 수 있음 (현재는 하위 8bit만 사용)")

    delta_u8 = addr_delta.astype(np.uint8)
    val_u8   = values.astype(np.uint8)

    # 기본 16bit 토큰 스트림
    word16 = (delta_u8.astype(np.uint16) << 8) | val_u8.astype(np.uint16)
    word16 = word16.astype(np.uint16)

    if not add_header:
        return word16

    # ---- 여기서부터 헤더 추가 ----
    # payload 길이: 토큰의 총 바이트 수 (uint16 개수 * 2)
    payload_bytes = word16.size * 2

    payload_low  = payload_bytes & 0xFFFF
    payload_high = (payload_bytes >> 16) & 0xFFFF


    rows, cols = shape
    rows = int(rows)
    cols = int(cols)

    if rows < 0 or rows > 0xFFFF or cols < 0 or cols > 0xFFFF:
        raise ValueError("rows, cols는 16bit 범위(0~65535) 안이어야 합니다.")

    # save_uint16_array_to_hex_chunked()에서 reversed(block)을 쓰는 걸 감안하면,
    # 내부 배열 순서는 [low, high, rows, cols]로 두고,
    # 하드웨어/파서에서 그에 맞게 해석하면 된다.
    header = np.array([payload_low, payload_high,
                       rows, cols], dtype=np.uint16)

    stream_with_header = np.concatenate([header, word16])
    return stream_with_header



def to_csc(B):
    """
    B: 2D numpy array (uint8)
    CSC 포맷으로 변환:
      V: non-zero 값 (uint8)
      R: 해당 값의 row index (uint16)
      C: 각 column 시작 index 포인터 (uint32), 길이 = n_cols + 1
    """
    B = np.asarray(B)
    m, n = B.shape

    values = []
    rows = []
    col_ptr = [0]  # 첫 번째 컬럼 시작 index = 0

    for j in range(n):
        col = B[:, j]
        nz_rows = np.nonzero(col)[0]
        values.extend(col[nz_rows])
        rows.extend(nz_rows)
        col_ptr.append(len(values))

    V = np.array(values, dtype=np.uint8)
    R = np.array(rows,   dtype=np.uint16)
    C = np.array(col_ptr, dtype=np.uint32)
    return V, R, C

def compute_ema_dense(B):
    """
    Dense 행렬 B(uint8)를 16bit bus로 읽는다고 가정.
    uint8 2개 = 16bit 1 word.
    """
    num_bytes = B.size  # uint8 → 1 byte
    ema_words = math.ceil(num_bytes / 2.0)
    return ema_words


def compute_ema_csc(V, R, C):
    """
    CSC(V,R,C)에서 16bit word 읽기 개수.
    가정:
      - V : uint8  → 2개당 1 word
      - R : uint16 → 1개당 1 word
      - C : uint32 → 1개당 2 word
    """
    # V: uint8 → 2개당 1 word
    ema_V = math.ceil(V.size / 2.0)

    # R: uint16 → 1개당 1 word
    ema_R = R.size

    # C: uint32 → 1개당 2 word
    ema_C = C.size * 2

    return ema_V + ema_R + ema_C


def compute_ema_delta(B_packed):
    """
    Delta 포맷(16bit uint16 배열)을 16bit bus로 읽는 경우,
    원소 하나당 16bit 1 word.
    """
    return B_packed.size*12/16


