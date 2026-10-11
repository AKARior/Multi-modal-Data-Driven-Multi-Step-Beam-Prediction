# Prepared Dataset Download

The prepared inference tensors are distributed separately because they are large. Download them here:

**[Quark Cloud: prepared model data](https://pan.quark.cn/s/d981b19a886d)**

After downloading, copy the contents into this directory so that the layout is:

```text
dataset/
├── 10to5/
│   ├── inputs.pt
│   ├── rgb.pt
│   └── metadata.json
└── 5to2/
    ├── inputs.pt
    ├── rgb.pt
    └── metadata.json
```

## Tensor contents

| Folder | Numeric tensor | RGB tensor per sample | Target tensor |
|---|---|---|---|
| `10to5/` | `[10, 5]` | `[10, 3, 128, 128]` | `[5]` |
| `5to2/` | `[5, 5]` | `[5, 3, 224, 224]` | `[2]` |

Run inference from the repository root with:

```bash
python eval/infer.py --protocol 10to5 --mode all
python eval/infer.py --protocol 5to2 --mode all
```
