import torch
import sys
import os

print(f"Python: {sys.version}")
print(f"Torch: {torch.__version__}")

# 1. Test DirectML
print("\n--- Testing DirectML ---")
try:
    import torch_directml
    dml = torch_directml.device()
    print(f"DirectML Device found: {dml}")
    
    # Simple tensor op
    x = torch.ones(5).to(dml)
    print(f"Tensor on DML: {x}")
    print("DirectML Basic Tensor Op: SUCCESS")
except ImportError:
    print("torch_directml not installed.")
except Exception as e:
    print(f"DirectML Error: {e}")

# 2. Test Model Architecture
print("\n--- Testing EML-NET Model ---")
try:
    from eml_net_model import EMLNet
    model = EMLNet()
    print("Model initialized successfully.")
    
    # Forward pass (CPU)
    dummy_input = torch.randn(2, 3, 480, 640)
    print(f"Running forward pass with input {dummy_input.shape}...")
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print("Model Forward Pass: SUCCESS")
except Exception as e:
    print(f"Model Error: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Test Complete ---")
