"""
src/evaluation/inference_cm5.py

Measures real-time inference latency of VAD models on CM5 hardware.
Run this DIRECTLY on CM5, not on LG Gram or MSI.

Requirements on CM5:
    pip install onnxruntime numpy

Usage:
    python inference_cm5.py --model_dir ~/vad_models/

Output:
    Prints latency table for paper
    Saves cm5_latency_results.csv
"""

import argparse
import time
import os
import json
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("Install onnxruntime: pip install onnxruntime")
    exit(1)


def measure_latency(session, n_warmup=20, n_runs=200):
    """
    Measure inference latency over n_runs iterations.
    Returns mean and std in milliseconds.
    """
    dummy = np.random.randn(1, 256).astype(np.float32)
    input_name = session.get_inputs()[0].name

    # warmup
    for _ in range(n_warmup):
        session.run(None, {input_name: dummy})

    # measure
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        times.append((time.perf_counter() - t0) * 1000)

    return float(np.mean(times)), float(np.std(times))


def run_vad_on_audio(session, audio_samples, window_size=256,
                     stride=128, threshold=0.5):
    """
    Run VAD on a full audio stream.
    Returns list of (start_sample, end_sample, is_speech) tuples.
    """
    input_name = session.get_inputs()[0].name
    results    = []

    for start in range(0, len(audio_samples) - window_size, stride):
        window  = audio_samples[start:start + window_size].reshape(1, 256)
        logits  = session.run(None, {input_name: window})[0]
        probs   = softmax(logits[0])
        is_speech = probs[1] > threshold
        results.append((start, start + window_size, bool(is_speech)))

    return results


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='~/vad_models/',
                        help='Directory containing ONNX model files')
    parser.add_argument('--n_runs', type=int, default=200,
                        help='Number of inference runs for latency measurement')
    args = parser.parse_args()

    model_dir = os.path.expanduser(args.model_dir)

    models = [
        'CNN1D_best.onnx',
        'WaveNetSmall_best.onnx',
        'ECAPAVAD_best.onnx',
        'TransformerVAD_best.onnx',
    ]

    # real-time constraint: 256 samples at 8kHz = 32ms
    REALTIME_MS = 32.0

    print(f"\n{'='*60}")
    print(f"  CM5 VAD Inference Latency Measurement")
    print(f"  Model dir : {model_dir}")
    print(f"  Runs      : {args.n_runs}")
    print(f"  Real-time : {REALTIME_MS}ms (256 samples at 8kHz)")
    print(f"{'='*60}\n")

    results = []

    for model_file in models:
        model_path = os.path.join(model_dir, model_file)
        model_name = model_file.replace('_best.onnx', '')

        if not os.path.exists(model_path):
            print(f"  [SKIP] {model_name} not found")
            continue

        # load model
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4  # CM5 has 4 cores
        session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )

        # measure latency
        mean_ms, std_ms = measure_latency(session, n_runs=args.n_runs)

        # model size
        size_kb = os.path.getsize(model_path) / 1024

        realtime_ok = mean_ms < REALTIME_MS
        status      = "✓ REAL-TIME" if realtime_ok else "✗ TOO SLOW"

        print(f"  {model_name:15s}: {mean_ms:.3f} ± {std_ms:.3f} ms  "
              f"| {size_kb:.0f} KB  | {status}")

        results.append({
            'model':         model_name,
            'mean_ms':       round(mean_ms, 3),
            'std_ms':        round(std_ms,  3),
            'size_kb':       round(size_kb, 1),
            'realtime_ok':   realtime_ok,
            'realtime_limit': REALTIME_MS,
        })

    print(f"\n{'='*60}")
    print(f"  Real-time threshold: {REALTIME_MS}ms")
    print(f"  All models within real-time constraint: "
          f"{'YES' if all(r['realtime_ok'] for r in results) else 'NO'}")
    print(f"{'='*60}")

    # save results
    import csv
    out_path = os.path.join(model_dir, 'cm5_latency_results.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n  Saved: {out_path}")
    print(f"\n  Copy this file back to LG Gram:")
    print(f"  scp user@192.168.12.155:{out_path} outputs/evaluation/")


if __name__ == "__main__":
    main()