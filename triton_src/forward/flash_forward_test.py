import torch
from flash_forward_triton import attention_triton_launch
from samples.flash_forward_triton_ref import attention_triton_launch as attention_triton_launch_ref
from forward_torch import naive_forward_aux

def test_attention_implementations():
    """
    Comprehensive test cases with seeded random generation to test both the main implementation and reference implementation against the naive implementation.
    """
    # Set seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)


    # Test case 0: Small dimensions
    print("=== Test Case 0: Small dimensions ===")
    B, N, H, D = 1, 16, 1, 32
    test_attention(B, N, H, D)

    # Test case 1: Small dimensions
    print("=== Test Case 1: Small dimensions ===")
    B, N, H, D = 2, 16, 32, 64
    test_attention(B, N, H, D)
    
    # Test case 2: Medium dimensions
    print("\n=== Test Case 2: Medium dimensions ===")
    B, N, H, D = 1, 64, 64, 256
    test_attention(B, N, H, D)
    
    # Test case 3: Sequence length
    print("\n=== Test Case 3: Large test ===")
    B, N, H, D = 128, 256, 128, 64
    test_attention(B, N, H, D)


def test_attention(B, N, H, D):
    """
    Tests the attention implementations for given dimensions.
    """
    print(f"Testing with B={B}, N={N}, H={H}, D={D}")
    
    # Generate random QKV
    torch.manual_seed(42)
    QKV = torch.randn(B, N, 3, H, D, device='cuda', dtype=torch.float32)
    
    # Convert to float32 for naive implementation
    QKV_f32 = QKV.float()
    Q, K, V = QKV_f32.unbind(dim=2)
    
    # Naive implementation
    softmax_scale = 1.0 / (D ** 0.5)
    _, _, O_naive, L_naive, _ = naive_forward_aux(Q, K, V, softmax_scale)
    
    # # Evaluate ref implementation
    # try:
    #     O_triton_ref = attention_triton_launch_ref(QKV)
    #     max_diff_ref = torch.max(torch.abs(O_naive - O_triton_ref.float())).item()
    #     max_diff_ref_normed = torch.max(torch.abs(O_naive - O_triton_ref.float()) / O_naive.abs()).item()
        
    #     print(f"O_triton_ref")
    #     print(O_triton_ref[0, :4, 0, :4])
    #     print(f"O_naive")
    #     print(O_naive[0, :4, 0, :4])
        
    #     print(f"  Naive vs Triton (ref):  {torch.allclose(O_naive, O_triton_ref.float(), atol=1e-2, rtol=1e-2)}")
    #     print(f"  Max diff (naive vs ref):  {max_diff_ref:.6f}")
    #     print(f"  Max diff (naive vs ref) normalized:  {max_diff_ref_normed:.6f}")
    # except Exception as e:
    #     print(f"  Error in Triton (ref) implementation: {e}")


    # Evaluate main implementation
    try:
        O_triton, L_triton = attention_triton_launch(QKV)
        max_diff_ref = torch.max(torch.abs(O_naive - O_triton.float())).item()
        max_diff_ref_normed = torch.max(torch.abs(O_naive - O_triton.float()) / O_naive.abs()).item()
        
        max_diff_L = torch.max(torch.abs(L_naive - L_triton.float())).item()
        max_diff_L_normed = torch.max(torch.abs(L_naive - L_triton.float()) / L_naive.abs()).item()
        O_ratio = O_triton / O_naive
        L_ratio = L_triton / L_naive

        print(f"O statistics")
        O_matches = torch.allclose(O_naive, O_triton.float(), atol=5e-3, rtol=1e-3)
        print(f"  Naive vs Triton (main): {torch.allclose(O_naive, O_triton.float(), atol=5e-3, rtol=1e-3)}")
        print(f"  Max diff (naive vs ref):  {max_diff_ref:.6f}")
        print(f"  Max diff (naive vs ref) normalized:  {max_diff_ref_normed:.6f}")
        if not O_matches:
            print(f"O_triton")
            print(O_triton[0, :4, 0, :4])
            print(f"O_naive")
            print(O_naive[0, :4, 0, :4])
            # print(f"O_ratio")
            # print(O_ratio[0, :4, 0, :4])

        print(f"L statistics")
        L_matches = torch.allclose(L_naive, L_triton.float(), atol=5e-3, rtol=1e-3)
        print(f"  Naive vs Triton (main): {torch.allclose(L_naive, L_triton.float(), atol=5e-3, rtol=1e-3)}")            
        print(f"  Max diff (naive vs ref):  {max_diff_L:.6f}")
        print(f"  Max diff (naive vs ref) normalized:  {max_diff_L_normed:.6f}")
        if not L_matches:
            print(f"L_triton")
            print(L_triton[0, 0, :4])
            print(f"L_naive")
            print(L_naive[0, 0, :4])
            # print(f"L_ratio")
            # print(L_ratio[0, 0, :4])

        # print(f"  O_ratio: {O_ratio[0, :, 0, :4]}")
    except Exception as e:
        print(f"  Error in Triton implementation: {e}")


if __name__ == "__main__":
    test_attention_implementations()


