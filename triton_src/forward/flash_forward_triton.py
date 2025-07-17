import torch
import triton
import triton.language as tl
import math

"""
CUDA_VISIBLE_DEVICES=1 TRITON_PRINT_AUTOTUNING=1 python3 flash.py
"""


def get_cuda_autotune_config():
    return [
        triton.Config({'Br': 16, 'Bc': 16}, num_stages=4, num_warps=8),
    ]
@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['N_const', 'D_const'],
)
@triton.jit
def attention_triton(
    Q_start_ptr, K_start_ptr, V_start_ptr, 
    O_start_ptr, L_start_ptr, M_start_ptr,
    N_const: tl.constexpr, H_const, D_const: tl.constexpr, softmax_scale,
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

    L_ptr = L_start_ptr + batch_id * lm_batch_stride + head_id * lm_heads_stride
    M_ptr = M_start_ptr + batch_id * lm_batch_stride + head_id * lm_heads_stride

    # L_chunk_ptr = tl.make_block_ptr(
    #     base = L_ptr, shape = (N_const,), strides = (1,), offsets = (Tr_i * Br,), block_shape = (Br,), order = (0,)
    # )
    # M_chunk_ptr = tl.make_block_ptr(
    #     base = M_ptr, shape = (N_const,), strides = (1,), offsets = (Tr_i * Br,), block_shape = (Br,), order = (0,)
    # )


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

    Qi = tl.load(Q_block_ptr)
    Oi = tl.zeros((Br, D_const), dtype=tl.float32)
    li = tl.zeros((Br,), dtype=tl.float32)
    mi = tl.full((Br,), float('-inf'), dtype=tl.float32)

    Tc = tl.cdiv(N_const, Bc)
    for Tc_j in range(0, Tc):
        Kj = tl.load(K_block_ptr)
        Sij = tl.dot(Qi, tl.trans(Kj)) * softmax_scale
        mij = tl.max(Sij, axis=1)
        mi_new = tl.maximum(mi, mij)

        Pij = tl.exp(Sij - mi_new[:, None])
        li = tl.exp(mi - mi_new) * li + tl.sum(Pij, axis=1)


        # update Oi
        Vj = tl.load(V_block_ptr)
        Oi = tl.exp(mi - mi_new)[:, None] * Oi + tl.dot(Pij, Vj)


        mi = tl.maximum(mi, mi_new)


        # advance the block pointers
        K_block_ptr = tl.advance(K_block_ptr, (Bc, 0))
        V_block_ptr = tl.advance(V_block_ptr, (Bc, 0))
    

    Oi = Oi * (1 / li)[:, None]
    tl.store(O_block_ptr, Oi)







    



def attention_triton_launch(QKV):
    QKV = QKV.to(torch.float32)
    Q_tensor, K_tensor, V_tensor = QKV.unbind(dim=2)
    Q_tensor_cont, K_tensor_cont, V_tensor_cont = Q_tensor.contiguous(), K_tensor.contiguous(), V_tensor.contiguous()


    B_const, N_const, H_const, D_const = Q_tensor_cont.shape
    O_tensor = torch.zeros((B_const, N_const, H_const, D_const), dtype=QKV.dtype, device=QKV.device)
    L_tensor = torch.zeros((B_const, H_const, N_const), dtype=QKV.dtype, device=QKV.device)
    M_tensor = torch.ones((B_const, H_const, N_const), dtype=QKV.dtype, device=QKV.device) * float("-inf")

    softmax_score = 1 / math.sqrt(D_const)

    B_stride, N_stride, H_stride, _ = Q_tensor_cont.stride()
    lm_batch_stride, lm_heads_stride, _ = M_tensor.stride()
    
    def grid(META):
        Tr = triton.cdiv(N_const, META['Br'])
        return (B_const, H_const, Tr)
    attention_triton[grid](
        Q_tensor_cont, K_tensor_cont, V_tensor_cont, 
        O_tensor, L_tensor, M_tensor,
        N_const, H_const, D_const, softmax_score,
        B_stride, N_stride, H_stride,
        lm_batch_stride, lm_heads_stride,
    )
    
    return O_tensor.to(QKV.dtype)
    