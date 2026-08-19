# test_gpu.py - run this to check what works
import numpy as np

print("=== Testing OpenVINO (inference on Intel Arc) ===")
try:
    import onnxruntime as ort
    session = ort.InferenceSession(
        "outputs/onnx/CNN1D_best.onnx",
        providers=["OpenVINOExecutionProvider"],
        provider_options=[{"device_type": "GPU"}]
    )
    dummy = np.random.randn(1, 256).astype(np.float32)
    result = session.run(None, {"audio": dummy})
    print(f"OpenVINO GPU: WORKING - output shape {result[0].shape}")
except Exception as e:
    print(f"OpenVINO GPU failed: {e}")

print("\n=== Testing DirectML (training on Intel Arc) ===")
try:
    import torch
    import torch_directml
    device = torch_directml.device()
    x = torch.randn(8, 256).to(device)
    print(f"DirectML: WORKING - device={device}")
    print(f"Tensor on GPU: {x.shape}")
except Exception as e:
    print(f"DirectML failed: {e}")

print("\n=== Testing torchaudio fix ===")
try:
    import torchaudio
    print(f"torchaudio: OK - {torchaudio.__version__}")
except Exception as e:
    print(f"torchaudio: {e}")