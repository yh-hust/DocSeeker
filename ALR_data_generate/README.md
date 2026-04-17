# ALR Data Generation

This module builds the ALR-style training data used in Stage I by generating structured **Analysis–Localization–Reasoning** responses from raw multi-page DocVQA samples and then verifying and merging the results, matching the data distillation pipeline described for DocSeeker.

## What it does

It converts raw question-answer-document examples into ALR-formatted supervision with explicit evidence grounding, which is the core data used to inject the ALR reasoning paradigm during supervised fine-tuning.

## Main files

- `generator.py`: generates ALR responses from the raw dataset.
- `verification.py`: verifies generated samples and filters low-quality outputs.
- `merge.py`: merges verified outputs into the final dataset.
- `generate.sh`: example generation entry script.
- `verify_and_merge.sh`: example verification and merge entry script.

## How to run

Set the API credentials in your environment first:

```bash
export OPENAI_API_KEY=your_api_key
export OPENAI_BASE_URL=your_api_base_url
```

Generate data:

```bash
cd ALR_data_generate
bash generate.sh
```

Verify and merge the generated results:

```bash
cd ALR_data_generate
bash verify_and_merge.sh
```

## Notes

Update the dataset paths, image paths, and output paths in the shell scripts before running them in a new environment.
