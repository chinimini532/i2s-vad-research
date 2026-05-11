"""
src/evaluation/export_onnx.py

Exports all trained VAD models to ONNX format for CM5 deployment.
Run this on LG Gram or MSI before copying to CM5.

Usage:
    python src/evaluation/export_onnx.py

Output:
    outputs/onnx/CNN1D_best.onnx
    outputs/onnx/WaveNetSmall_best.onnx
    outputs/onnx/ECAPAVAD_best.onnx
    outputs/onnx/TransformerVAD_best.onnx
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.cnn1d           import CNN1D
from src.models.wavenet_small   import WaveNetSmall
from src.models.ecapa_vad       import ECAPAVAD
from src.models.transformer_vad import TransformerVAD

# use exp3 models (A-law trained) for deployment
# these are the best models for real telephony hardware
MODEL_DIR = ROOT / "outputs" / "exp3_alaw_musan" / "models"
ONNX_DIR  = ROOT / "outputs" / "onnx"
ONNX_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CLASSES = {
    "CNN1D":          CNN1D(num_classes=2),
    "WaveNetSmall":   WaveNetSmall(num_classes=2),
    "ECAPAVAD":       ECAPAVAD(num_classes=2),
    "TransformerVAD": TransformerVAD(num_classes=2),
}

def export_model(name, model, pt_path, onnx_path):
    """Export a single model to ONNX format."""
    print(f"\n  Exporting {name}...")

    # load trained weights
    ckpt = torch.load(str(pt_path), map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # dummy input — one window of 256 samples at 8kHz
    dummy_input = torch.randn(1, 256)

    # verify model works
    with torch.no_grad():
        out = model(dummy_input)
    print(f"    Input  shape: {dummy_input.shape}")
    print(f"    Output shape: {out.shape}")

    # export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['audio'],
        output_names=['logits'],
        dynamic_axes={
            'audio':  {0: 'batch_size'},
            'logits': {0: 'batch_size'},
        }
    )

    # verify ONNX model
    import onnx
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    size_kb = onnx_path.stat().st_size / 1024
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Saved : {onnx_path}")
    print(f"    Size  : {size_kb:.1f} KB")
    print(f"    Params: {n_params:,}")

    return size_kb


def main():
    print(f"\n{'='*55}")
    print(f"  ONNX Export for CM5 Deployment")
    print(f"  Source: {MODEL_DIR}")
    print(f"  Output: {ONNX_DIR}")
    print(f"{'='*55}")

    results = []
    for name, model in MODEL_CLASSES.items():
        pt_path   = MODEL_DIR / f"{name}_best.pt"
        onnx_path = ONNX_DIR  / f"{name}_best.onnx"

        if not pt_path.exists():
            print(f"\n  [SKIP] {name} - model not found: {pt_path}")
            continue

        size_kb = export_model(name, model, pt_path, onnx_path)
        results.append((name, size_kb))

    print(f"\n{'='*55}")
    print(f"  Export complete")
    print(f"{'='*55}")
    for name, size in results:
        print(f"  {name:15s}: {size:.1f} KB")

    print(f"\n  Copy outputs/onnx/ to CM5:")
    print(f"  scp -r outputs/onnx/ user@192.168.12.155:~/vad_models/")
    print(f"\n  Then run on CM5:")
    print(f"  python src/evaluation/inference_cm5.py")


if __name__ == "__main__":
    main()