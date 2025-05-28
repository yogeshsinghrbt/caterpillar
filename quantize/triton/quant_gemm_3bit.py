# Reference https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/awq_triton.py
# Copyright 2024 The vLLM team.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at 
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl

AWQ_TRITON_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]


@triton.jit
def awq_dequantize_kernel_3bit(
    qweight_ptr,  # quantized matrix
    scales_ptr,  # scales, per group
    zeros_ptr,  # zeros, per group
    group_size,  # Should always be one of the supported group sizes
    result_ptr,  # Output matrix
    num_cols,  # input num cols in qweight
    num_rows,  # input num rows in qweight
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    # Setup the pids.
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)

    # Compute offsets and masks for qweight_ptr.
    offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offsets = num_cols * offsets_y[:, None] + offsets_x[None, :]

    # print("offsets_x", offsets_x)
    # print("offsets_y", offsets_y)
    # print("offsets", offsets)

    masks_y = offsets_y < num_rows
    masks_x = offsets_x < num_cols

    masks = masks_y[:, None] & masks_x[None, :]

    # Compute offsets and masks for result output ptr.
    result_offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    result_offsets_x = pid_x * BLOCK_SIZE_X * 10 + tl.arange(0, BLOCK_SIZE_X * 16)
    result_offsets = (
        10 * num_cols * result_offsets_y[:, None] + result_offsets_x[None, :]
    )

    # print("result_offsets_x", result_offsets_x)
    # print("result_offsets_y", result_offsets_y)
    # print("result_offsets", result_offsets)

    result_masks_y = result_offsets_y < num_rows
    result_masks_x = (result_offsets_x < num_cols * 10) & (result_offsets_x <  (pid_x + 1) * 10)
    result_masks = result_masks_y[:, None] & result_masks_x[None, :]
    # 
    # print ("result_masks", result_masks.shape)
    # print ("result_masks_x", result_masks_x)
    

    # Load the weights.
    b = tl.load(qweight_ptr + offsets, masks)

    b = tl.join(b, b).reshape(b.shape[:-1] + [2 * b.shape[-1]])
    b = tl.join(b, b).reshape(b.shape[:-1] + [2 * b.shape[-1]])
    b = tl.join(b, b).reshape(b.shape[:-1] + [2 * b.shape[-1]])
    b = tl.join(b, b).reshape(b.shape[:-1] + [2 * b.shape[-1]])
    
    # iweights = tl.interleave(iweights, iweights)
    # iweights = tl.interleave(iweights, iweights)
    # iweights = tl.interleave(iweights, iweights)

    reverse_awq_order_tensor = ((tl.arange(0, 2) * 5)[None, :] + tl.arange(0, 8)[:, None]).reshape(16)

    # # Create the necessary shifts to use to unpack.
    shifts = reverse_awq_order_tensor * 3

    # if (pid_m == 0 and pid_n == 1) and pid_z == 1:
    #     print ("shifts", shifts)

    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_Y ,16 ))

    # Unpack and reorder: shift out the correct 4-bit value and mask.
    b = (b >> shifts) & 0x7

    # Compute zero offsets and masks.
    zero_offsets_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(0, 1)
    zero_offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    zero_offsets = num_cols * zero_offsets_y[:, None] + zero_offsets_x[None, :]

    zero_masks_y = zero_offsets_y < num_rows // group_size
    zero_masks_x = zero_offsets_x < num_cols
    zero_masks = zero_masks_y[:, None] & zero_masks_x[None, :]

    # Load the zeros.
    zeros = tl.load(zeros_ptr + zero_offsets, zero_masks)
    
    # zeros = tl.interleave(zeros, zeros)
    # zeros = tl.interleave(zeros, zeros)
    # zeros = tl.interleave(zeros, zeros)

    zeros =  tl.join(zeros, zeros).reshape(zeros.shape[:-1] + [2 * zeros.shape[-1]])
    zeros = tl.join(zeros, zeros).reshape(zeros.shape[:-1] + [2 * zeros.shape[-1]])
    zeros = tl.join(zeros, zeros).reshape(zeros.shape[:-1] + [2 * zeros.shape[-1]])
    zeros = tl.join(zeros, zeros).reshape(zeros.shape[:-1] + [2 * zeros.shape[-1]])
    
    zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 16))

    # Unpack and reorder: shift out the correct 4-bit value and mask.
    zeros = (zeros >> shifts) & 0x7

    # Compute scale offsets and masks.
    scale_offsets_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(0, 1)
    scale_offsets_x = pid_x * BLOCK_SIZE_X * 10 + tl.arange(0, BLOCK_SIZE_X * 16)
    scale_offsets = num_cols * 10 * scale_offsets_y[:, None] + scale_offsets_x[None, :]
    scale_masks_y = scale_offsets_y < num_rows // group_size
    scale_masks_x = scale_offsets_x < num_cols * 10
    scale_masks = scale_masks_y[:, None] & scale_masks_x[None, :]

    # Load the scales.
    scales = tl.load(scales_ptr + scale_offsets, scale_masks)
    scales = tl.broadcast_to(scales, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 16))

    # Dequantize.
    b = (b - zeros) * scales
    b = b.to(result_ptr.type.element_ty)

    # Finally, store.
    # print ("result_masks", result_masks)
    # print ("result_offsets", result_offsets)
    # print ("b", b)
    tl.store(result_ptr + result_offsets, b, result_masks)


# qweights - [K     , M // 8], int32
# scales   - [K // G, M     ], float16
# zeros    - [K // G, M // 8], int32
def awq_dequantize_triton_3bit(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    block_size_x: int = 1,
    block_size_y: int = 128,
) -> torch.Tensor:

    # print ("-----", "qweight", qweight.shape)
    # print ("-----", "qzeros", qzeros.shape)
    # print ("-----", "scales", scales.shape)
    
    K = qweight.shape[0]
    M = scales.shape[1]
    group_size = qweight.shape[0] // scales.shape[0]

    assert K > 0 and M > 0
    assert scales.shape[0] == K // group_size and scales.shape[1] == M
    assert qzeros.shape[0] == K // group_size and qzeros.shape[1] == M // 10
    assert group_size <= K
    assert group_size in AWQ_TRITON_SUPPORTED_GROUP_SIZES or group_size == K

    # Result tensor:
    # number of rows = same as input tensor
    # number of cols = 8 x input tensor num cols
    result = torch.empty(
        qweight.shape[0],
        scales.shape[1],
        device=qweight.device,
        dtype=scales.dtype,
    )

    X = qweight.shape[1]  # num cols
    Y = qweight.shape[0]  # num rows
    

    grid = lambda META: (
        triton.cdiv(X, META["BLOCK_SIZE_X"]),
        triton.cdiv(Y, META["BLOCK_SIZE_Y"]),
    )

    #grid = lambda META: (2, 1)

    awq_dequantize_kernel_3bit[grid](
        qweight,
        scales,
        qzeros,
        group_size,
        result,
        X,
        Y,
        BLOCK_SIZE_X=block_size_x,
        BLOCK_SIZE_Y=block_size_y,
    )

    return result[:, :scales.shape[1] - (scales.shape[1] % 128)]


@triton.jit
def awq_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    zeros_ptr,
    scales_ptr,
    M,
    N,
    K,
    group_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)

    num_pid_n = (N + 10 - 1) // 10

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    

    
    accumulator_dtype = c_ptr.type.element_ty

    # NOTE: This doesn't work in TRITON_INTERPRET=1 mode.  Use below instead.
    # accumulator = tl.arange(0, BLOCK_SIZE_N)
    # accumulator = tl.broadcast_to(accumulator[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N))
    # accumulator = accumulator & 0x0
    # accumulator = accumulator.to(accumulator_dtype)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype) #gpu

    # Create reverse AWQ order as tensor: [0, 4, 1, 5, 2, 6, 3, 7]
    # that will map given indices to the correct order.
    #reverse_awq_order_tensor = tl.arange(0, 16)

    reverse_awq_order_tensor = ((tl.arange(0, 2) * 5)[None, :] + tl.arange(0, 8)[:, None]).reshape(16)

    # # Create the necessary shifts to use to unpack.
    shifts = reverse_awq_order_tensor * 3

    # if (pid_m == 0 and pid_n == 1) and pid_z == 1:
    #     print ("shifts", shifts)

    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_K ,16 ))

    # Offsets and masks.
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    masks_am = offsets_am < M

    offsets_bn = (pid_n * (BLOCK_SIZE_N // 10) ) + tl.arange(0, 1)
    masks_bn = offsets_bn <  ((N + 10 - 1 )// 10)

    offsets_zn = pid_n * (BLOCK_SIZE_N // 10) + tl.arange(0, 1)
    masks_zn = offsets_zn <  ((N + 10 - 1 )// 10)

    offsets_sn = pid_n * 10 + tl.arange(0, BLOCK_SIZE_N)
    masks_sn = offsets_sn < N

    offsets_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    
    offsets_a = K * offsets_am[:, None] + offsets_k[None, :]
    offsets_b = ((N + 10 - 1 )// 10) * offsets_k[:, None] + offsets_bn[None, :]
    
    a_ptrs = a_ptr + offsets_a
    b_ptrs = b_ptr + offsets_b

    # NOTE: Use this in TRITON_INTERPRET=1 mode instead of tl.cdiv
    block_offset = BLOCK_SIZE_K * SPLIT_K
    for k in range(0, (K + block_offset - 1) // (block_offset)):
    #for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        masks_k = offsets_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        a = tl.load(a_ptrs, mask=masks_a)

        masks_b = masks_k[:, None] & masks_bn[None, :]
        b = tl.load(b_ptrs, mask=masks_b)



        # if (pid_m == 0 and pid_n == 0) and (pid_z == 0 and k == 0):
        #     print ("---------------k b",k, b, b.shape )

        # b = tl.join(b, b).reshape(b.shape[:-1] + [2 * b.shape[-1]])
        # b = tl.join(b, b).reshape(b.shape[:-1] + [2 * b.shape[-1]])
        # b = tl.join(b, b).reshape(b.shape[:-1] + [2 * b.shape[-1]])
        # b = tl.join(b, b).reshape(b.shape[:-1] + [2 * b.shape[-1]])

        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)

        # if (pid_m == 0 and pid_n == 0) and (pid_z == 0 and k == 0):
        #     print ("-----", "b", b.shape)


        # Dequantize b.
        offsets_szk = (
            BLOCK_SIZE_K * SPLIT_K * k + pid_z * BLOCK_SIZE_K
        ) // group_size + tl.arange(0, 1)

            
        offsets_z =((N + 10 - 1 )// 10) * offsets_szk[:, None] + offsets_zn[None, :]
            
        masks_zk = offsets_szk < K // group_size
        masks_z = masks_zk[:, None] & masks_zn[None, :]
        zeros_ptrs = zeros_ptr + offsets_z
        zeros = tl.load(zeros_ptrs, mask=masks_z)
        
        # zeros =  tl.join(zeros, zeros).reshape(zeros.shape[:-1] + [2 * zeros.shape[-1]])
        # zeros = tl.join(zeros, zeros).reshape(zeros.shape[:-1] + [2 * zeros.shape[-1]])
        # zeros = tl.join(zeros, zeros).reshape(zeros.shape[:-1] + [2 * zeros.shape[-1]])
        # zeros = tl.join(zeros, zeros).reshape(zeros.shape[:-1] + [2 * zeros.shape[-1]])
        
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        
        zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        offsets_s = (((N + 10 - 1 )// 10) * 10) * offsets_szk[:, None] + offsets_sn[None, :]
        
        masks_sk = offsets_szk < K // group_size
        masks_s = masks_sk[:, None] & masks_sn[None, :]
        scales_ptrs = scales_ptr + offsets_s
        scales = tl.load(scales_ptrs, mask=masks_s)
        
        scales = tl.broadcast_to(scales, (BLOCK_SIZE_K, BLOCK_SIZE_N))
            
        b = (b >> shifts) & 0x7
        zeros = (zeros >> shifts) & 0x7
        b = (b - zeros) * scales
        
        b = b.to(c_ptr.type.element_ty)


        accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)
        

        offsets_k += BLOCK_SIZE_K * SPLIT_K
        a_ptrs += BLOCK_SIZE_K * SPLIT_K
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * ((N + 10 - 1 )// 10)

    c = accumulator.to(c_ptr.type.element_ty)
    
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * 10 + tl.arange(0, BLOCK_SIZE_N)
        
    c_ptrs = c_ptr + N * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & ((offs_cn[None, :] < ((pid_n * 10) + 10)) & (offs_cn[None, :] < N))

    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)
    



# split_k_iters - parallelism along K-dimension, int, power of 2.
def awq_quant_gemm_triton_3bit(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    split_k_iters: int,
    block_size_m: int = 32,
    block_size_n: int = 16,
    block_size_k: int = 128,
) -> torch.Tensor:
    
    M, K = input.shape
    N = scales.shape[1] - (scales.shape[1] % 128)
    group_size = 128
    

    grid = lambda META: (
        triton.cdiv(M, 32) * triton.cdiv(N, 10),
        split_k_iters,
    )


    result = torch.zeros((M, N), dtype=scales.dtype, device=input.device)


    # A = input, B = qweight, C = result
    # A = M x K, B = K x N, C = M x N
    awq_gemm_kernel[grid](
        input,
        qweight,
        result,
        qzeros,
        scales,
        M,
        N,
        K,
        group_size,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        SPLIT_K=split_k_iters,
    )

    return result