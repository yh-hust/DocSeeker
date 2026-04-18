# Evidence-aware GRPO

This module implements Stage II reinforcement learning with Evidence-aware GRPO, which further improves both evidence localization and final answer accuracy after supervised fine-tuning.

## What it does

It optimizes the policy model with a reward that jointly considers output format, evidence-page localization, and answer correctness, corresponding to the EviGRPO stage in DocSeeker.

## Main files

- `scripts/run_EviGRPO.sh`: main entry script for EviGRPO training.
- `verl/`: local training framework code and protocol/config helpers used by the PPO/GRPO pipeline.

## How to run

### Parepare training data

Download online filter training data and validation data from [Link](https://pan.baidu.com/s/16n4ZR_pHGEvP6YPbOr8UNw?pwd=yy8i)

### Start Fine-Tuning

```bash
cd EviGRPO
bash scripts/run_EviGRPO.sh
```

## Notes

Before running, update the training files, validation files, image path, and base model/checkpoint path in `scripts/run_EviGRPO.sh`.
