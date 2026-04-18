# Supervised Fine-Tuning

This module performs Stage I supervised fine-tuning on the distilled ALR data so the backbone model learns the page-aware **Analysis–Localization–Reasoning** workflow proposed in DocSeeker.

## What it does

It fine-tunes Qwen2.5-VL on ALR-formatted multi-page document data, which teaches the model to analyze the question, localize supporting pages, and produce grounded answers.

## Main files

- `scripts/finetune_on_ALR_data.sh`: main entry script for ALR-data SFT.
- `scripts/zero2.json`, `scripts/zero3.json`, `scripts/zero3_offload.json`: DeepSpeed configuration files.
- `qwenvl/`: training code and model-side utilities used by the fine-tuning pipeline.

## How to run

Parepare training data:

Download ALR training data from [Link](https://pan.baidu.com/s/19r9JdHCryP0LF15n1fO4Pg？pwd=mq9m)

Run the following command to start the training process on your GPU cluster:

```bash
cd Finetune
bash scripts/finetune_on_ALR_data.sh
```

## Notes

Before running, adjust the model path, dataset identifiers, and output directory in `scripts/finetune_on_ALR_data.sh` to match your local training environment.
