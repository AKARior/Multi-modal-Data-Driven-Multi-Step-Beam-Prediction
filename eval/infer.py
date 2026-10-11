"""Run numeric, image or multimodal prediction from published .pt inputs."""
import argparse
import json
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from inference import build_predictor


class PreparedInputs(Dataset):
    """Read prepared tensors only; no raw-data processing or dataset splitting."""

    def __init__(self, folder, mode="multi", limit=None):
        folder = Path(folder)
        self.metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        self.inputs = torch.load(folder / "inputs.pt", map_location="cpu", weights_only=True)
        self.mode = mode
        samples = len(self.inputs["numeric"])
        seq_len, pred_len = self.metadata["input_len"], self.metadata["pred_len"]
        if samples != self.metadata["samples"] or samples < 1:
            raise ValueError("Invalid sample count")
        if tuple(self.inputs["numeric"].shape[1:]) != (seq_len, 5):
            raise ValueError("Invalid numeric input shape")
        if self.inputs["image_slots"].shape != self.inputs["numeric"].shape[:2]:
            raise ValueError("Invalid image slot shape")
        if "targets" in self.inputs and tuple(self.inputs["targets"].shape) != (samples, pred_len):
            raise ValueError("Invalid target shape")
        if limit is not None and limit < 1:
            raise ValueError("--limit must be positive")
        self.count = min(samples, limit) if limit is not None else samples
        self.rgb = None
        if mode != "numeric":
            self.rgb = torch.load(folder / "rgb.pt", map_location="cpu", weights_only=True)
            if tuple(self.rgb.shape) != (self.metadata["frames"], *self.metadata["image_shape"]):
                raise ValueError("Invalid RGB input shape")
            slots = self.inputs["image_slots"]
            if slots.min().item() < 0 or slots.max().item() >= len(self.rgb):
                raise ValueError("Image slot is outside the published RGB tensor")

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        numeric = self.inputs["numeric"][index].float()
        image = torch.empty(0)
        if self.mode != "numeric":
            image = self.rgb[self.inputs["image_slots"][index]].float()
        target = self.inputs["targets"][index].long() if "targets" in self.inputs else torch.empty(0, dtype=torch.long)
        return numeric, image, target


@torch.inference_mode()
def evaluate(args, mode):
    folder = Path(args.data_dir) if args.data_dir else ROOT / "dataset" / args.protocol
    dataset = PreparedInputs(folder, mode, args.limit)
    if dataset.metadata["protocol"] != args.protocol:
        raise ValueError("Input data protocol differs from checkpoint protocol")
    reference = json.loads((ROOT / "weights" / args.protocol / "model_config.json").read_text(encoding="utf-8"))
    batch_size = args.batch_size or reference["batch_sizes"][mode]
    model = build_predictor(args.protocol, mode, args.device, args.checkpoint)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    totals = {k: 0 for k in (1, 2, 3, 5)}
    batch_sums = {k: 0.0 for k in totals}
    prediction_example = None
    tokens = batches = 0
    loss_sum = batch_loss = 0.0
    for numeric, image, target in loader:
        numeric = numeric.to(args.device) if mode != "image" else None
        image = image.to(args.device) if mode != "numeric" else None
        logits = model(numeric=numeric, image=image, mode=mode)
        if tuple(logits.shape[1:]) != (dataset.metadata["pred_len"], 64):
            raise ValueError("Unexpected model output shape")
        indices = logits.topk(5, dim=-1).indices
        if prediction_example is None:
            prediction_example = indices[0].cpu().tolist()
        if target.numel():
            target = target.to(args.device)
            correct = indices.eq(target.unsqueeze(-1))
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 64), target.reshape(-1)).item()
            for k in totals:
                hit = correct[..., :k].any(-1)
                totals[k] += hit.sum().item()
                batch_sums[k] += hit.float().mean().item()
            loss_sum += loss * target.numel()
            batch_loss += loss
            tokens += target.numel()
        batches += 1
    weighted = {f"top{k}": totals[k] / tokens for k in totals} if tokens else None
    averaged = {f"top{k}": batch_sums[k] / batches for k in totals} if tokens else None
    aggregation = reference["metric_aggregation"][mode]
    metrics = averaged if aggregation == "batch_mean" else weighted
    loss = (batch_loss / batches if aggregation == "batch_mean" else loss_sum / tokens) if tokens else None
    result = {"protocol": args.protocol, "mode": mode, "samples": len(dataset), "prediction_tokens": tokens,
              "batch_size": batch_size, "metric_aggregation": aggregation, "metrics": metrics,
              "loss": loss, "token_weighted_metrics": weighted, "batch_mean_metrics": averaged,
              "first_sample_top5_beams": prediction_example,
              "scope": "subset" if args.limit and args.limit < dataset.metadata["samples"] else "full_prepared_test_inputs"}
    if args.output_dir:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / f"{mode}_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=["10to5", "5to2"], required=True)
    parser.add_argument("--mode", choices=["numeric", "image", "multi", "all"], default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit", type=int, help="Optional quick inference subset")
    parser.add_argument("--data-dir")
    parser.add_argument("--checkpoint", help="Override checkpoint for a single mode")
    parser.add_argument("--output-dir", help="Optionally save JSON metrics; default prints only")
    args = parser.parse_args()
    if args.checkpoint and args.mode == "all":
        parser.error("--checkpoint requires a single --mode")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for mode in (["numeric", "image", "multi"] if args.mode == "all" else [args.mode]):
        evaluate(args, mode)


if __name__ == "__main__":
    main()
