<!-- omit in toc -->
DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding (CVPR 2026 Highlight)
================================================================
Hao Yan, Yuliang Liu, Xingchen Liu, Yuyi Zhang, Minghui Liao, Jihao Wu, Wei Chen, Xiang Bai

<h5 align="center">

[![arXiv](https://img.shields.io/badge/ArXiv-XXXX.XXXXX-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/XXXX.XXXXX)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://your-project-page.github.io/)
[![Code](https://img.shields.io/badge/Code-Coming_Soon-white)](./README.md)
[![Model](https://img.shields.io/badge/Model-Coming_Soon-gray)]()
<br>
</h5>


## News

* `2026.x.x` 📄 Paper released on arXiv.
* `2026.xs.x` 🚀 Code / model / demo will be released soon.
* `2026.4.9` 🔥 **DocSeeker** is accepted to **CVPR 2026** as a **Highlight** paper.


---

## Introduction

**DocSeeker** is a multimodal large language model for **long document understanding**. Existing MLLMs often struggle as document length grows, because crucial evidence is easily buried in many irrelevant pages, while most training data only provides short final answers without explicit evidence grounding.

To address this, DocSeeker introduces a structured **Analysis–Localization–Reasoning (ALR)** paradigm, which encourages the model to first analyze the question, then localize evidence pages, and finally perform grounded reasoning before generating the answer. Built on **Qwen-2.5-VL-7B-Instruct**, DocSeeker further combines **ALR CoT distillation**, **Evidence-aware GRPO**, and **Evidence-Guided Resolution Allocation (EGRA)** for effective long-document training. This leads to strong gains on both in-domain and out-of-domain benchmarks, while making the reasoning process more interpretable and evidence-grounded.

---

## Highlights

- **A new structured reasoning paradigm for long documents.**  
  We propose **ALR (Analysis–Localization–Reasoning)**, which turns long-document QA from direct answer prediction into an explicit evidence-grounded reasoning process.

- **Evidence-grounded training instead of answer-only supervision.**  
  DocSeeker is trained with a two-stage pipeline that combines **high-quality ALR CoT distillation** and **Evidence-aware GRPO**, explicitly optimizing both **evidence localization** and **answer correctness**.

- **Efficient long-document learning with EGRA.**  
  We introduce **Evidence-Guided Resolution Allocation (EGRA)**, which preserves high resolution for evidence pages while reducing redundant cost on non-evidence pages, enabling more effective and scalable training on long visual documents.

---

## Main Results

DocSeeker achieves strong performance across both in-domain and out-of-domain long document benchmarks, outperforming representative open-source long-document MLLMs and remaining competitive with strong closed-source models.

<p align="center">
    <img src="assets/performance.png" width="1000"/>
<p>

DocSeeker sets strong results on **DUDE**, **MP-DocVQA**, and **SlideVQA**, and substantially outperforms representative open-source methods on challenging long-document benchmarks such as **MMLongBench-doc** and **LongDocURL**.

---


## Environment

```bash
conda create -n docseeker python=3.10 -y
conda activate docseeker

pip install -r requirements.txt

# Optional but commonly needed in our setup
python -c "from opencv_fixer import AutoFix; AutoFix()"
```

---

## Implementation

Data distillation
```bash
cd ALR_data_generate

# Set API credentials
export OPENAI_API_KEY=your_api_key
export OPENAI_BASE_URL=your_api_base_url

# Edit the paths and index range in generate.sh before running:
# - INPUT_DATA
# - IMAGE_DIR
# - SAVE_BASE_DIR
# - START_IDX / END_IDX
# - MODEL_NAME

bash generate.sh

# Verify and merge the generated samples
bash verify_and_merge.sh
```
---

Supervised Fine-tuning
```bash
cd Finetune

# Edit the paths and experiment settings in:
# scripts/finetune_on_ALR_data.sh
# - MODEL_PATH
# - DATASETS
# - OUTPUT_DIR

bash scripts/finetune_on_ALR_data.sh
```

Evi-GRPO
```bash
cd EviGRPO

# Edit the paths in:
# scripts/run_EviGRPO.sh
# - TRAIN_FILES
# - VAL_FILES
# - IMAGE_PATH
# - MODEL_PATH

bash scripts/run_EviGRPO.sh
```

Evaluation
```bash
cd Evaluation

# Set API credentials if answer extraction / metric parsing requires them
export OPENAI_API_KEY=your_api_key
export OPENAI_BASE_URL=your_api_base_url

# Edit the evaluation scripts before running:
# - DEVICE
# - MODEL_PATH
# - EXPERIMENT_NAME
# - checkpoint_id
# - dataset-specific paths if needed

# MP-DocVQA / DUDE / SlideVQA
bash scripts/run_mpdocvqa_dude_sft.sh

# MMLongBench-doc / LongDocURL
bash scripts/run_mmlongben_longdocurl.sh
```

---

## Citation

```bibtex
@inproceedings{yan2026docseeker,
  title     = {DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding},
  author    = {Hao Yan and Yuliang Liu and Xingchen Liu and Yuyi Zhang and Minghui Liao and Jihao Wu and Wei Chen and Xiang Bai},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2026}
}
```

---
## Acknowledgement
Our work benefit from the following open-source projects:
- [Qwen2.5 VL](https://github.com/QwenLM/qwen-code)
- [verl](https://github.com/volcengine/verl)


## Contact

For questions and collaborations, please contact the authors of the paper.
