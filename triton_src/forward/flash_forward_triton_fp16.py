import torch
import triton
import triton.language as tl
import math

"""
CUDA_VISIBLE_DEVICES=1 TRITON_PRINT_AUTOTUNING=1 python3 flash.py
"""

target_dtype = tl.bfloat16

def get_cuda_autotune_config():
    return [
        triton.Config({'Br': 64, 'Bc': 64}, num_stages=3, num_warps=4),
        triton.Config({'Br': 128, 'Bc': 64}, num_stages=2, num_warps=4),
        triton.Config({'Br': 64, 'Bc': 128}, num_stages=2, num_warps=4),
        triton.Config({'Br': 64, 'Bc': 64}, num_stages=4, num_warps=8),
        triton.Config({'Br': 128, 'Bc': 64}, num_stages=4, num_warps=8),
        triton.Config({'Br': 64, 'Bc': 128}, num_stages=4, num_warps=8),
        triton.Config({'Br': 128, 'Bc': 128}, num_stages=4, num_warps=8),
        triton.Config({'Br': 128, 'Bc': 128}, num_stages=4, num_warps=8),
        triton.Config({'Br': 128, 'Bc': 128}, num_stages=4, num_warps=16),
        triton.Config({'Br': 128, 'Bc': 128}, num_stages=4, num_warps=16),
    ]
# def get_cuda_autotune_config():
#     num_stages_list = [4,8]
#     num_warps_list = [8, 16]
#     Br_list = [128, 256]
#     Bc_list = [128, 256]
#     configs = []
#     for num_stages in num_stages_list:
#         for num_warps in num_warps_list:
#             for Br in Br_list:
#                 for Bc in Bc_list:
#                     configs.append(triton.Config({'Br': Br, 'Bc': Bc}, num_stages=num_stages, num_warps=num_warps))
#     return configs
@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['N_const', 'D_const'],
)
@triton.jit
def attention_triton(
    Q_start_ptr, K_start_ptr, V_start_ptr, 
    O_start_ptr, L_start_ptr,
    N_const: tl.constexpr, D_const: tl.constexpr, softmax_scale,
    B_stride, N_stride, H_stride,
    lm_batch_stride, lm_heads_stride,
    Br : tl.constexpr, Bc : tl.constexpr):

    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    Tr_i = tl.program_id(2)

    # point to (batch_id, 0, head_id, 0)
    Q_ptr = Q_start_ptr + batch_id * B_stride + head_id * H_stride
    O_ptr = O_start_ptr + batch_id * B_stride + head_id * H_stride

    K_ptr = K_start_ptr + batch_id * B_stride + head_id * H_stride
    V_ptr = V_start_ptr + batch_id * B_stride + head_id * H_stride

    Q_block_ptr = tl.make_block_ptr(
        base = Q_ptr, shape = (N_const, D_const), strides = (N_stride, 1), 
        offsets = (Tr_i * Br, 0), block_shape = (Br, D_const), order = (1, 0)
    )
    O_block_ptr = tl.make_block_ptr(
        base = O_ptr, shape = (N_const, D_const), strides = (N_stride, 1),
        offsets = (Tr_i * Br, 0), block_shape = (Br, D_const), order = (1, 0)
    )

    K_block_ptr = tl.make_block_ptr(
        base = K_ptr, shape = (N_const, D_const), strides = (N_stride, 1), 
        offsets = (0, 0), block_shape = (Bc, D_const), order = (1, 0)
    )
    V_block_ptr = tl.make_block_ptr(
        base = V_ptr, shape = (N_const, D_const), strides = (N_stride, 1), 
        offsets = (0, 0), block_shape = (Bc, D_const), order = (1, 0)
    )

    Qi = tl.load(Q_block_ptr, boundary_check=(0,), padding_option = "zero")
    Oi = tl.zeros((Br, D_const), dtype=tl.float32)
    li = tl.zeros((Br,), dtype=tl.float32)
    mi = tl.full((Br,), float('-inf'), dtype=tl.float32)

    Tc = tl.cdiv(N_const, Bc)
    for Tc_j in range(0, Tc):
        Tc_indexes = Tc_j * Bc + tl.arange(0, Bc)
        Tc_mask = Tc_indexes < N_const
        
        Kj = tl.load(K_block_ptr, boundary_check=(0,), padding_option = "zero")
        Sij = tl.dot(Qi, tl.trans(Kj)) * softmax_scale
        mij = tl.max(Sij, axis=1)
        mi_new = tl.maximum(mi, mij)

        Pij = tl.exp(Sij - mi_new[:, None]) * Tc_mask[None, :]
        Pij = Pij.to(target_dtype)
        li = tl.exp(mi - mi_new) * li + tl.sum(Pij, axis=1)


        # update Oi
        Vj = tl.load(V_block_ptr, boundary_check=(0,), padding_option = "zero")
        Oi = tl.exp(mi - mi_new)[:, None] * Oi 
        Oi = tl.dot(Pij, Vj, Oi)


        mi = tl.maximum(mi, mi_new)


        # advance the block pointers
        K_block_ptr = tl.advance(K_block_ptr, (Bc, 0))
        V_block_ptr = tl.advance(V_block_ptr, (Bc, 0))
    

    Oi = Oi * (1 / li)[:, None]
    Oi = Oi.to(target_dtype)
    tl.store(O_block_ptr, Oi, boundary_check = (0,))


    L_ptr = L_start_ptr + batch_id * lm_batch_stride + head_id * lm_heads_stride
    L_chunk_ptr = tl.make_block_ptr(
        base = L_ptr, shape = (N_const,), strides = (1,), offsets = (Tr_i * Br,), block_shape = (Br,), order = (0,)
    )
    Li = tl.log(li) + mi
    Li = Li.to(target_dtype)
    tl.store(L_chunk_ptr, Li, boundary_check = (0,))


def attention_triton_launch(QKV):
    Q_tensor, K_tensor, V_tensor = QKV.unbind(dim=2)
    Q_tensor_cont, K_tensor_cont, V_tensor_cont = Q_tensor.contiguous(), K_tensor.contiguous(), V_tensor.contiguous()


    B_const, N_const, H_const, D_const = Q_tensor_cont.shape
    O_tensor = torch.zeros((B_const, N_const, H_const, D_const), dtype=QKV.dtype, device=QKV.device)
    L_tensor = torch.zeros((B_const, H_const, N_const), dtype=QKV.dtype, device=QKV.device)

    softmax_score = float(1 / math.sqrt(D_const))

    B_stride, N_stride, H_stride, _ = Q_tensor_cont.stride()
    lm_batch_stride, lm_heads_stride, _ = L_tensor.stride()
    
    def grid(META):
        Tr = triton.cdiv(N_const, META['Br'])
        return (B_const, H_const, Tr)
    attention_triton[grid](
        Q_tensor_cont, K_tensor_cont, V_tensor_cont, 
        O_tensor, L_tensor,
        N_const, D_const, softmax_score,
        B_stride, N_stride, H_stride,
        lm_batch_stride, lm_heads_stride,
    )
    torch.cuda.synchronize()
    return O_tensor.to(QKV.dtype), L_tensor.to(QKV.dtype)
    