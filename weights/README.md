# Inference Checkpoint Download

The proposed-model checkpoints are distributed separately because they are large. Download them here:

**[Quark Cloud: model weights](https://pan.quark.cn/s/09fc39e19bfc)**

After downloading, copy the contents into this directory so that the layout is:

```text
weights/
├── 10to5/
│   ├── model_config.json
│   ├── uav_side_numeric_beam_prediction_ckpt.pth
│   ├── edge_side_image_beam_prediction_ckpt.pth
│   └── edge_multimodal_beam_prediction_ckpt.pth
└── 5to2/
    ├── model_config.json
    ├── uav_side_numeric_beam_prediction_ckpt.pth
    ├── edge_side_image_beam_prediction_ckpt.pth
    └── edge_multimodal_beam_prediction_ckpt.pth
```

The numeric and image checkpoint files contain the tensors required by their active prediction path. The multimodal checkpoint contains the complete proposed model state. `model_config.json` selects the protocol-specific sequence length, prediction length, encoder settings and decoder settings; always use the configuration under the same protocol folder as the checkpoint.

Once the files are placed, run:

```bash
python eval/infer.py --protocol 10to5 --mode all
python eval/infer.py --protocol 5to2 --mode all
```

Do not use a 10-to-5 checkpoint with the 5-to-2 configuration or vice versa. The loader checks checkpoint keys and tensor shapes before inference.
