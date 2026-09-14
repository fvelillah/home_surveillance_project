#!/usr/bin/env python3
"""
Model Exporter for Edge Surveillance Analytics.

Exports a lightweight MobileNetV3 model to ONNX format for on-device edge feature extraction.
Produces an ONNX model that takes normalized video frames and outputs dense feature embeddings.

Usage:
    python scripts/export_edge_model.py --output data/models/mobilenet_v3_small.onnx --dim 512
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_model")


def export_mobilenet_v3_onnx(output_path: Path, embedding_dim: int = 512) -> bool:
    """Export MobileNetV3 feature extractor to ONNX using PyTorch."""
    try:
        import torch
        import torch.nn as nn
        import torchvision.models as models
    except ImportError:
        logger.error("PyTorch and Torchvision are strictly required for edge model operations.")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Instantiating MobileNetV3-Small backbone from torchvision...")

    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    backbone = models.mobilenet_v3_small(weights=weights)

    # Replace classifier with a projection to embedding_dim
    in_features = backbone.classifier[0].in_features
    
    class FeatureExtractorWrapper(nn.Module):
        def __init__(self, base_model, in_dim, out_dim):
            super().__init__()
            self.features = base_model.features
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.projection = nn.Linear(in_dim, out_dim)

        def forward(self, x):
            # x shape: (B, 3, 224, 224)
            feat = self.features(x)
            feat = self.avgpool(feat)
            feat = torch.flatten(feat, 1)
            out = self.projection(feat)
            # L2 normalize
            out = nn.functional.normalize(out, p=2, dim=1)
            return out

    model = FeatureExtractorWrapper(backbone, in_features, embedding_dim)
    model.eval()

    dummy_input = torch.randn(16, 3, 224, 224, requires_grad=False)
    logger.info("Exporting model to ONNX: %s", output_path)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["frames"],
        output_names=["embeddings"],
        dynamic_axes={
            "frames": {0: "batch_size"},
            "embeddings": {0: "batch_size"},
        },
    )

    logger.info("ONNX model exported successfully to %s (size: %.2f MB)", output_path, output_path.stat().st_size / (1024 * 1024))
    return True


def main():
    parser = argparse.ArgumentParser(description="Export edge video feature extractor to ONNX")
    parser.add_argument("--output", type=Path, default=Path("data/models/mobilenet_v3_small.onnx"), help="Target ONNX file path")
    parser.add_argument("--dim", type=int, default=512, help="Output embedding dimensionality")
    args = parser.parse_args()

    success = export_mobilenet_v3_onnx(args.output, args.dim)
    if not success:
        sys.exit(0)  # Clean exit with advisory


if __name__ == "__main__":
    main()
