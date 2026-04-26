"""Profile model parameters, approximate FLOPs/MACs, latency and CUDA memory.

This script is intended to support the computational-complexity evidence
requested in revision. It profiles TSD-NET and several simple local baselines.
For external baselines such as Informer, TimeXer and GBT, add their classes to
MODEL_REGISTRY and keep the same input/output interface: [B, T, F] -> [B, H, 1].
"""
import argparse
import os
import sys
import time

import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.stack import Stack
from baselines.simple_baselines import CNNForecaster, LSTMForecaster, TransformerForecaster


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def profile_latency(model, x, runs=50, warmup=10):
    device = x.device
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / runs
        peak_mem = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    return dt * 1000, peak_mem / (1024 ** 2)


def build_models(input_size, input_len, horizon, num_blocks, no_profiler, channels):
    return {
        "TSD-NET": Stack(
            input_size=input_size,
            encoder_channels=channels,
            input_seqlen=input_len,
            forecast_seqlen=horizon,
            num_blocks=num_blocks,
            enable_profiler=not no_profiler,
        ),
        "LSTM": LSTMForecaster(input_size=input_size, horizon=horizon),
        "CNN": CNNForecaster(input_size=input_size, horizon=horizon),
        "Transformer": TransformerForecaster(input_size=input_size, horizon=horizon),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-size", type=int, default=2)
    parser.add_argument("--input-len", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--channels", default="16,32,64,64", help="TSD-NET encoder channels, e.g. 16,32,64,64")
    parser.add_argument("--no-profiler", action="store_true")
    parser.add_argument("--out", default="results/complexity_report.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(args.batch_size, args.input_len, args.input_size, device=device)
    channels = [int(v) for v in args.channels.split(",") if v.strip()]
    rows = []
    for name, model in build_models(args.input_size, args.input_len, args.horizon, args.num_blocks, args.no_profiler, channels).items():
        model = model.to(device)
        latency_ms, peak_mem_mb = profile_latency(model, x, runs=args.runs)
        macs = None
        profiler = getattr(model, "profiler", None)
        if profiler is not None:
            df = profiler.summarize()
            macs = int(df["MACs(approx)"].sum()) if "MACs(approx)" in df else None
        rows.append({
            "Model": name,
            "Params": count_params(model),
            "MACs_or_FLOPs_approx": macs,
            "Latency_ms_batch1": latency_ms,
            "PeakMem_MB": peak_mem_mb,
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
