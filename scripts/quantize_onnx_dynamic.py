"""
Post-training dynamic quantization for ONNX model.

Usage:
  python scripts/quantize_onnx_dynamic.py --input models/bge-m3-ko-onnx/model.onnx --output models/bge-m3-ko-onnx/model.int8.onnx
"""
import argparse
from pathlib import Path

from onnxruntime.quantization import quantize_dynamic, QuantType


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=str(args.input),
        model_output=str(args.output),
        optimize_model=True,
        per_channel=False,
        reduce_range=False,
        weight_type=QuantType.QInt8,
    )
    print(f"Wrote quantized model: {args.output}")


if __name__ == "__main__":
    main()


