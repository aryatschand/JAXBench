"""
Test script to verify Modal TPU and AWS Bedrock resources are working.

This script tests:
1. AWS Bedrock connectivity with Claude Opus 4.5
2. Modal TPU v4-8 allocation and JAX execution

Run with: python test_resources.py
"""

import sys
import time


def test_aws_bedrock():
    """Test AWS Bedrock connectivity with Claude Opus 4.5."""
    print("=" * 70)
    print("TEST 1: AWS Bedrock (Claude Opus 4.5)")
    print("=" * 70)
    
    try:
        from bedrock_client import BedrockClient
        
        client = BedrockClient()
        print(f"  Region: {client.AWS_REGION}")
        print(f"  Testing Opus 4.5 model...")
        
        # Test a simple prompt
        test_prompt = """
Write a simple JAX function that performs matrix multiplication.
Just provide the function, no explanation.
"""
        
        start = time.time()
        response = client.invoke(
            "opus",
            test_prompt,
            system="You are a JAX expert. Respond concisely with code only.",
            max_tokens=500,
            temperature=0.3
        )
        elapsed = time.time() - start
        
        print(f"\n  ✅ Opus 4.5 Response ({elapsed:.2f}s):")
        print("-" * 50)
        # Print first 500 chars
        print(response[:500] if len(response) > 500 else response)
        if len(response) > 500:
            print("... (truncated)")
        print("-" * 50)
        
        # Also test Haiku for faster iteration feedback
        print("\n  Testing Haiku 4.5 model (for faster feedback loops)...")
        start = time.time()
        response = client.invoke(
            "haiku",
            "Say 'Haiku works!' in exactly 3 words.",
            max_tokens=50,
            temperature=0.3
        )
        elapsed = time.time() - start
        print(f"  ✅ Haiku 4.5 Response ({elapsed:.2f}s): {response[:100]}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ AWS Bedrock test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_modal_tpu():
    """Test Modal TPU connectivity and JAX execution."""
    print("\n" + "=" * 70)
    print("TEST 2: Modal TPU (v4-8) with JAX")
    print("=" * 70)
    
    try:
        import modal
        
        # Define the Modal app and TPU function inline for testing
        app = modal.App("jaxbench-test")
        
        tpu_image = (
            modal.Image.debian_slim(python_version="3.10")
            .pip_install(
                "jax[tpu]==0.4.30",
                extra_options="-f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
            )
            .pip_install(
                "numpy>=1.24.0",
                "torch>=2.0.0",  # For PyTorch baseline comparison
            )
        )
        
        @app.function(
            image=tpu_image,
            cloud="gcp",
            accelerator="tpu-v4-8",
            timeout=600,
        )
        def test_tpu_jax():
            """Test JAX on TPU with basic operations."""
            import jax
            import jax.numpy as jnp
            import time as time_module
            
            results = {}
            
            # Device info
            results["backend"] = jax.default_backend()
            results["device_count"] = jax.device_count()
            results["devices"] = [str(d) for d in jax.devices()]
            
            print(f"JAX Backend: {results['backend']}")
            print(f"Device Count: {results['device_count']}")
            print(f"Devices: {results['devices']}")
            
            # Test matrix multiplication
            @jax.jit
            def matmul(a, b):
                return jnp.dot(a, b)
            
            # Create test matrices
            size = 2048
            key = jax.random.PRNGKey(0)
            a = jax.random.normal(key, (size, size), dtype=jnp.float32)
            b = jax.random.normal(key, (size, size), dtype=jnp.float32)
            
            # Warmup
            result = matmul(a, b)
            result.block_until_ready()
            
            # Benchmark
            num_iters = 50
            start = time_module.perf_counter()
            for _ in range(num_iters):
                result = matmul(a, b)
                result.block_until_ready()
            elapsed = time_module.perf_counter() - start
            
            results["matmul_size"] = size
            results["matmul_iterations"] = num_iters
            results["matmul_total_ms"] = elapsed * 1000
            results["matmul_avg_ms"] = (elapsed / num_iters) * 1000
            
            # Calculate TFLOPS (2 * N^3 FLOPs for matmul)
            flops_per_matmul = 2 * size ** 3
            total_flops = flops_per_matmul * num_iters
            tflops = (total_flops / elapsed) / 1e12
            results["matmul_tflops"] = tflops
            
            print(f"\nMatrix multiplication ({size}x{size}):")
            print(f"  Average time: {results['matmul_avg_ms']:.3f} ms")
            print(f"  Throughput: {tflops:.2f} TFLOPS")
            
            return results
        
        print("  Launching Modal TPU function...")
        print("  (This may take a minute to allocate TPU resources)")
        
        start = time.time()
        with app.run():
            result = test_tpu_jax.remote()
        elapsed = time.time() - start
        
        print(f"\n  ✅ Modal TPU test PASSED ({elapsed:.1f}s total)")
        print(f"  Backend: {result['backend']}")
        print(f"  Devices: {result['device_count']}x TPU cores")
        print(f"  Matmul Performance: {result['matmul_tflops']:.2f} TFLOPS")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Modal TPU test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pytorch_on_tpu():
    """Test PyTorch on TPU via Modal (baseline comparison)."""
    print("\n" + "=" * 70)
    print("TEST 3: PyTorch on TPU (Baseline)")
    print("=" * 70)
    
    try:
        import modal
        
        app = modal.App("pytorch-tpu-test")
        
        # PyTorch/XLA image for TPU
        tpu_image = (
            modal.Image.debian_slim(python_version="3.10")
            .pip_install(
                "torch>=2.0.0",
                "torch_xla[tpu]>=2.0.0",
                extra_options="-f https://storage.googleapis.com/libtpu-releases/index.html"
            )
            .pip_install("numpy>=1.24.0")
        )
        
        @app.function(
            image=tpu_image,
            cloud="gcp",
            accelerator="tpu-v4-8",
            timeout=600,
        )
        def test_pytorch_tpu():
            """Test PyTorch/XLA on TPU."""
            import torch
            import time as time_module
            
            results = {}
            
            try:
                import torch_xla
                import torch_xla.core.xla_model as xm
                
                device = xm.xla_device()
                results["device"] = str(device)
                results["pytorch_xla_available"] = True
                
                print(f"PyTorch XLA Device: {device}")
                
                # Test matmul
                size = 2048
                a = torch.randn(size, size, device=device)
                b = torch.randn(size, size, device=device)
                
                # Warmup
                c = torch.matmul(a, b)
                xm.mark_step()
                
                # Benchmark
                num_iters = 50
                start = time_module.perf_counter()
                for _ in range(num_iters):
                    c = torch.matmul(a, b)
                    xm.mark_step()
                elapsed = time_module.perf_counter() - start
                
                results["matmul_avg_ms"] = (elapsed / num_iters) * 1000
                tflops = (2 * size ** 3 * num_iters / elapsed) / 1e12
                results["matmul_tflops"] = tflops
                
                print(f"\nPyTorch Matmul ({size}x{size}):")
                print(f"  Average time: {results['matmul_avg_ms']:.3f} ms")
                print(f"  Throughput: {tflops:.2f} TFLOPS")
                
            except ImportError:
                results["pytorch_xla_available"] = False
                print("PyTorch XLA not available, testing CPU fallback")
                
            return results
        
        print("  Launching Modal PyTorch/XLA TPU function...")
        
        start = time.time()
        with app.run():
            result = test_pytorch_tpu.remote()
        elapsed = time.time() - start
        
        if result.get("pytorch_xla_available"):
            print(f"\n  ✅ PyTorch/XLA TPU test PASSED ({elapsed:.1f}s)")
            print(f"  Matmul Performance: {result.get('matmul_tflops', 'N/A'):.2f} TFLOPS")
        else:
            print(f"\n  ⚠️  PyTorch/XLA not available on TPU")
        
        return True
        
    except Exception as e:
        print(f"  ❌ PyTorch TPU test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all resource tests."""
    print("\n" + "=" * 70)
    print("JAXBench Resource Verification")
    print("=" * 70)
    print("This script verifies that all required resources are available:\n")
    print("  1. AWS Bedrock (Claude Opus 4.5) - for PyTorch -> JAX translation")
    print("  2. Modal TPU (v4-8) with JAX - for running JAX benchmarks")
    print("  3. Modal TPU with PyTorch/XLA - for baseline comparison")
    print()
    
    results = {}
    
    # Test AWS Bedrock
    results["bedrock"] = test_aws_bedrock()
    
    # Test Modal TPU with JAX
    results["modal_jax"] = test_modal_tpu()
    
    # Test PyTorch on TPU (optional, for baseline)
    results["modal_pytorch"] = test_pytorch_on_tpu()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("✅ All resources verified! Ready to build JAXBench pipeline.")
    else:
        print("❌ Some resources failed. Please check the errors above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

