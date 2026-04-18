# Evaluation

This module runs DocSeeker inference and benchmark evaluation on multi-page document understanding datasets, covering the in-domain and out-of-domain settings reported in the paper.

## What it does

It performs model inference on benchmark datasets and then computes answer extraction and evaluation metrics for datasets such as MP-DocVQA, DUDE, SlideVQA, MMLongBench-Doc, and LongDocURL.

## Main files

- `vlm_inference_on_mpdoc.py`: multi-page VLM inference entry point.
- `llm_cal.py`: answer extraction and post-processing.
- `metric_cal.py`: metric aggregation and evaluation.
- `scripts/run_mpdocvqa_dude_sft.sh`: example evaluation script for MP-DocVQA, DUDE, and SlideVQA.
- `scripts/run_mmlongben_longdocurl.sh`: example evaluation script for MMLongBench-Doc and LongDocURL.

## How to run

## How to run

prepare the images for each benchmark and place them into their respective folders under the `tmp` directory:

```text
Evaluation/
├── mmlongbench_doc/
│   └── tmp/
│       ├── document_1_page_1.jpg
│       ├── document_1_page_2.jpg
│       ├── ...
│       └── document_N_page_M.jpg
├── dude/
│   └── tmp/
│       ├── ...
```

Set the API credentials used by the answer-extraction step if needed:

```bash
export OPENAI_API_KEY=your_api_key
export OPENAI_BASE_URL=your_api_base_url
```

Run one of the provided scripts:

```bash
cd Evaluation
bash scripts/run_evaluation.sh
```
## Notes

Before running, update the checkpoint path, dataset list, visible devices, and output settings in the corresponding shell script.
