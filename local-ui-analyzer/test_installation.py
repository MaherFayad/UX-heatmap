import torch
import sys
import os

print(f"Python: {sys.version}")
print(f"Torch: {torch.__version__}")
try:
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Capability: {torch.cuda.get_device_capability(0)}")
    else:
        print("WARNING: CUDA is NOT available.")
except Exception as e:
    print(f"Error checking CUDA: {e}")

# 1. Test Tensor Ops (CUDA)
print("\n--- Testing CUDA Tensor Ops ---")
try:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        x = torch.ones(5).to(device)
        y = x * 2
        print(f"Tensor on CUDA: {y}")
        print("Basic Tensor Op: SUCCESS")
    else:
        print("Skipping CUDA tensor op test (CUDA not available)")
except Exception as e:
    print(f"CUDA Op Error: {e}")

# 2. Test Model Architecture
print("\n--- Testing EML-NET Model ---")
try:
    from eml_net_model import EMLNet
    model = EMLNet()
    print("Model initialized successfully.")
    
    # Forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Moving model to {device}...")
    model.to(device)
    
    dummy_input = torch.randn(2, 3, 480, 640).to(device)
    print(f"Running forward pass with input {dummy_input.shape}...")
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print("Model Forward Pass: SUCCESS")
except Exception as e:
    print(f"Model Error: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Test Complete ---")
