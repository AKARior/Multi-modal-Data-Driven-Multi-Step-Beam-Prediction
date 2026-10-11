"""Construct and strictly load the published prediction paths."""
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
FILES = {"numeric": "uav_side_numeric_beam_prediction_ckpt.pth",
         "image": "edge_side_image_beam_prediction_ckpt.pth",
         "multi": "edge_multimodal_beam_prediction_ckpt.pth"}


def build_predictor(protocol="10to5", mode="multi", device="cuda", weight_path=None):
    if protocol not in {"10to5", "5to2"} or mode not in FILES:
        raise ValueError("Use protocol 10to5/5to2 and mode numeric/image/multi")
    try:
        from .model import LiteMultiBeamModel
    except ImportError:
        from model import LiteMultiBeamModel
    folder = ROOT / "weights" / protocol
    config = json.loads((folder / "model_config.json").read_text(encoding="utf-8"))
    model = LiteMultiBeamModel(**config["model_kwargs"])
    path = Path(weight_path) if weight_path else folder / FILES[mode]
    state = torch.load(path, map_location="cpu", weights_only=True)
    expected = model.state_dict()
    prefixes = {"numeric": ("numeric.", "decoder."), "image": ("vision.", "decoder.")}.get(mode)
    if prefixes:
        expected = {key: value for key, value in expected.items() if key.startswith(prefixes)}
    if set(expected) != set(state):
        raise ValueError(f"Invalid checkpoint keys: missing={set(expected)-set(state)}, extra={set(state)-set(expected)}")
    for key, value in expected.items():
        if state[key].shape != value.shape:
            raise ValueError(f"Checkpoint tensor shape differs: {key}")
    model.load_state_dict(state, strict=prefixes is None)
    return model.to(device).eval()
