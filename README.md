# DocSeeker
<p align="center">
    <!-- You can replace this with your teaser / framework figure -->
    <img src="assets/teaser.png" width="900"/>
<p>

<h3 align="center"> <a href="https://arxiv.org/abs/XXXX.XXXXX">DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding</a></h3>
<h2 align="center">CVPR 2026 Highlight</h2>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/ArXiv-XXXX.XXXXX-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/XXXX.XXXXX)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://your-project-page.github.io/)
[![Code](https://img.shields.io/badge/Code-Coming_Soon-white)](./README.md)
[![Model](https://img.shields.io/badge/Model-Coming_Soon-gray)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](./LICENSE)
<br>
</h5>

> [**[CVPR 2026 Highlight] DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding**](https://arxiv.org/abs/XXXX.XXXXX)  
> Hao Yan, Yuliang Liu*, Xingchen Liu, Yuyi Zhang, Minghui Liao, Jihao Wu, Wei Chen*, Xiang Bai  
> Huazhong University of Science and Technology, Huawei Inc.  
> \* Corresponding authors

---

## News

* `2026.x.x` 🔥 **DocSeeker** is accepted to **CVPR 2026 as a Highlight paper**.
* `2026.x.x` 📄 Paper released on arXiv.
* `2026.x.x` 🚀 Code / model / demo will be released soon.

---

## Introduction

**DocSeeker** is a document multimodal large language model for **long document understanding**, built to address the severe performance degradation of existing MLLMs on multi-page documents as document length grows. The paper identifies two core bottlenecks in long-document reasoning: **low signal-to-noise ratio**, where crucial evidence is buried in many irrelevant pages, and **supervision scarcity**, where training data often provides only short final answers without intermediate reasoning or evidence grounding. :contentReference[oaicite:2]{index=2}

To solve these issues, DocSeeker introduces a structured **Analysis–Localization–Reasoning (ALR)** paradigm. Instead of directly predicting answers, the model first analyzes the question, then explicitly localizes evidence pages, and finally performs grounded reasoning before outputting the answer together with evidence page IDs. This design improves interpretability and strengthens the model’s ability to reason over long, visually rich documents. :contentReference[oaicite:3]{index=3}

DocSeeker is built on top of **Qwen-2.5-VL-7B-Instruct** and is trained with a two-stage pipeline:  
1. **SFT with distilled ALR CoT data**, to teach the model the ALR reasoning pattern;  
2. **Evidence-aware GRPO (EviGRPO)**, to jointly optimize answer correctness and evidence localization.  

The paper also proposes **Evidence-Guided Resolution Allocation (EGRA)**, which keeps evidence pages at high resolution while downsampling many non-evidence pages during training, making long-document learning more efficient and effective. 

---

## Highlights

- **Structured visual reasoning for long documents.**  
  We propose the **ALR paradigm**, which decomposes document reasoning into **question analysis**, **evidence localization**, and **reasoning process**, with explicit evidence grounding. :contentReference[oaicite:5]{index=5}

- **Two-stage training for grounded reasoning.**  
  DocSeeker combines **high-quality ALR CoT distillation** and **Evidence-aware GRPO** to improve both localization and reasoning. 

- **Efficient long-context visual training.**  
  We introduce **EGRA**, a simple yet effective strategy that allocates different image resolutions to evidence and non-evidence pages during training. :contentReference[oaicite:7]{index=7}

- **Strong generalization to ultra-long documents.**  
  Even though training uses relatively short multi-page datasets, DocSeeker generalizes robustly to much longer out-of-domain documents. 

- **Naturally synergistic with visual RAG.**  
  The explicit evidence-localization capability makes DocSeeker a strong foundation for building robust retrieval-augmented document reasoning systems. :contentReference[oaicite:9]{index=9}

---

## Main Results

DocSeeker achieves strong performance across both in-domain and out-of-domain long document benchmarks. According to the paper, it outperforms the same-architecture baseline by **30%–60%** across five document VQA benchmarks, and establishes open-source state-of-the-art performance on several out-of-domain settings. 

| Method | DUDE | MP-DocVQA | MMLongBench-doc | LongDocURL | SlideVQA |
|--------|------|-----------|-----------------|------------|----------|
| Baseline | 35.2 | 70.1 | 25.4 | 37.8 | 59.8 |
| Baseline-SFT (short-answer) | 56.0 | 82.9 | 28.8 | 42.7 | 67.4 |
| DocSeeker-SFT | 56.8 | 82.1 | 38.6 | 49.1 | 75.2 |
| **DocSeeker** | **57.4** | **86.2** | **40.1** | **51.7** | **77.1** |

These results show that the gains are not only from more training, but specifically from the proposed **ALR reasoning paradigm** and **evidence-aware optimization**. 

---

## Framework Overview

DocSeeker follows an **Analyze–Locate–Reason** workflow:

1. **Question Analysis**: understand what the user is asking.
2. **Evidence Localization**: identify which pages contain useful evidence.
3. **Reasoning Process**: synthesize grounded evidence into the final answer.
4. **Answer Output**: return both the answer and the supporting evidence page IDs.

This structured formulation improves interpretability and helps the model distinguish useful evidence from noise in long document inputs. :contentReference[oaicite:12]{index=12}

---

## 🐳 Model Zoo

| Model | Backbone | Status |
|-------|----------|--------|
| DocSeeker | Qwen-2.5-VL-7B-Instruct | Coming Soon |

---

## Environment

```bash
# Coming soon
