def test_numerical_stability():
    """
    Tests numerical stability with extreme values.
    """
    print("\n=== Numerical Stability Test ===")
    torch.manual_seed(42)
    
    # Test with very large values
    B, N, H, D = 1, 168, 256, 512
    QKV = torch.randn(B, N, H, D, device='cuda', dtype=torch.bfloat16) * 1e9
    
    try:
        O_triton = attention_triton_launch(QKV)
        O_triton_ref = attention_triton_launch_ref(QKV)
        
        # Check for NaN or inf
        has_nan_main = torch.isnan(O_triton).any().item()
        has_inf_main = torch.isinf(O_triton).any().item()
        has_nan_ref = torch.isnan(O_triton_ref).any().item()
        has_inf_ref = torch.isinf(O_triton_ref).any().item()
        
        print(f"  Main implementation - NaN: {has_nan_main}, Inf: {has_inf_main}")
        print(f"  Ref implementation  - NaN: {has_nan_ref}, Inf: {has_inf_ref}")
        
    except Exception as e:
        print(f"  Error in stability test: {e}")

def test_performance():
    """
    Tests performance with larger inputs.
    """
    print("\n=== Performance Test ===")
    torch.manual_seed(42)
    
    # Larger test case
    B, N, H, D = 2, 256, 64, 1024
    QKV = torch.randn(B, N, H, D, device='cuda', dtype=torch.bfloat16)
    
    import time
    
    # Warm up
    for _ in range(3):
        _ = attention_triton_launch_ref(QKV)
    
    # Time reference implementation
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(10):
        O_ref = attention_triton_launch_ref(QKV)
    torch.cuda.synchronize()
    ref_time = (time.time() - start_time) / 10
    # Time main implementation
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(10):
        O_main = attention_triton_launch(QKV)
    torch.cuda.synchronize()
    main_time = (time.time() - start_time) / 10
    
    print(f"  Reference implementation: {ref_time:.4f}s")
    print(f"  Main implementation:   {main_time:.4f}s")
    print(f"  Speedup: {main_time/ref_time:0.2f}x")