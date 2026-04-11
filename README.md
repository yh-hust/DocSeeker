# DocSeeker

<p align="center">
    <!-- Replace with your teaser / framework figure -->
    <img src="assets/teaser.png" width="900"/>
<p>

<h3 align="center">
    <a href="https://arxiv.org/abs/XXXX.XXXXX">DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding</a>
</h3>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/ArXiv-XXXX.XXXXX-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/XXXX.XXXXX)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://your-project-page.github.io/)
[![Code](https://img.shields.io/badge/Code-Coming_Soon-white)](./README.md)
[![Model](https://img.shields.io/badge/Model-Coming_Soon-gray)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](./LICENSE)
<br>
</h5>

> [**[CVPR 2026 Highlight] DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding**](https://arxiv.org/abs/XXXX.XXXXX)  
> Hao Yan, Yuliang Liu, Xingchen Liu, Yuyi Zhang, Minghui Liao, Jihao Wu, Wei Chen, Xiang Bai

---

## News

* `2026.x.x` 📄 Paper released on arXiv.
* `2026.x.x` 🚀 Code / model / demo will be released soon.
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

| Method | DUDE | MP-DocVQA | MMLongBench-doc | LongDocURL | SlideVQA |
|--------|------|-----------|-----------------|------------|----------|
| mPLUG-DocOwl2 | 46.8 | 69.4 | 13.4 | 5.3 | - |
| M3DocRAG | - | 84.4 | 21.0 | 35.1 | 55.7 |
| Vis-RAG | - | 70.9 | 18.8 | 41.9 | 50.7 |
| VDocRAG | 44.0 | 62.6 | 18.4 | 39.8 | 42.0 |
| InternVL3 | 47.4 | 80.8 | 24.1 | 38.7 | 54.4 |
| GPT-4o | 54.1 | 67.4 | **42.8** | **64.5** | - |
| **DocSeeker** | **57.9** | **86.0** | 40.1 | 51.7 | **77.8** |

DocSeeker sets strong results on **DUDE**, **MP-DocVQA**, and **SlideVQA**, and substantially outperforms representative open-source methods on challenging long-document benchmarks such as **MMLongBench-doc** and **LongDocURL**.

---

## 🐳 Model Zoo

| Model | Backbone | Status |
|-------|----------|--------|
| DocSeeker | Qwen-2.5-VL-7B-Instruct | Coming Soon |

---

## Environment

```bash
# Coming soon
```

---

## Training

```bash
# Coming soon
```

---

## Inference

```bash
# Coming soon
```

---

## Dataset

Our training pipeline is built upon existing multi-page document VQA datasets, including **MP-DocVQA** and **DUDE**, and evaluation is conducted on both in-domain and out-of-domain benchmarks such as **MMLongBench-doc**, **LongDocURL**, and **SlideVQA**.

More details about data preparation, distillation, and filtering will be released soon.

---

## Evaluation

```bash
# Coming soon
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

This project will release more details after publication. We sincerely thank the open-source multimodal and document understanding community for their valuable contributions.

---


## Contact

For questions and collaborations, please contact the authors of the paper.
