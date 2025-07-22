import torch
import math


def naive_forward(Q, K, V, softmax_scale):
    """
    args:
        - Q: (B,N,H,d)
        - K: (B,N,H,d)
        - V: (B,N,H,d)
        - softmax_scale: float
    returns:
        - S: (B,H,N,N)
        - P: (B,H,N,N)
        - O: (B,H,N,d)
    """
    S = torch.einsum("b t h d, b s h d -> b h t s", Q, K) * softmax_scale
    P = torch.softmax(S, dim=-1)  # softmax along key dimension
    O = torch.einsum("b h t s, b s h d -> b t h d", P, V)
    return S, P, O


def naive_forward_aux(Q, K, V, softmax_scale):
    """
    args:
        - Q: (B,N,H,d)
        - K: (B,N,H,d)
        - V: (B,N,H,d)
        - softmax_scale: float
    returns:
        - S: (B,H,N,N)
        - P: (B,H,N,N)
        - O: (B,H,N,d)
        - L: (B,H,N)
        - M: (B,H,N)

    """
    S = torch.einsum("b t h d, b s h d -> b h t s", Q, K) * softmax_scale
    P = torch.softmax(S, dim=-1)  # softmax along key dimension
    O = torch.einsum("b h t s, b s h d -> b t h d", P, V)

    M, _ = torch.max(S, dim=-1)

    # the first is strictly correct, but the second is what flash 2 actually uses
    # L = torch.sum(torch.exp(S - M[:, :, :, None]), dim = -1)
    L = torch.log(torch.sum(torch.exp(S), dim = -1))


    return S, P, O, L, M

def naive_forward_aux_wrapper(QKV):
    QKV = QKV.to(torch.float32)
    Q, K, V = QKV.unbind(dim=2)
    B, N, H, D = Q.shape
    
    # Naive implementation
    softmax_scale = 1.0 / (D ** 0.5)
    S_naive, P_naive, O_naive, L_naive, M_naive = naive_forward_aux(Q, K, V, softmax_scale)
    return S_naive, P_naive, O_naive, L_naive, M_naive