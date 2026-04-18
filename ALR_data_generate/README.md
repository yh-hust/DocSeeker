# ALR Data Generation

This module builds the ALR-style training data used in Stage I by generating structured **Analysis–Localization–Reasoning (ALR)** responses from raw multi-page DocVQA samples and then verifying the results, matching the data distillation pipeline (data generation and secondary verification) described for DocSeeker.

## Main files

- `generator.py`: generates ALR responses from the raw dataset.
- `verification.py`: verifies generated samples and filters low-quality outputs.
- `merge.py`: merges verified outputs into the final dataset.
- `generate.sh`: example generation entry script.
- `verify_and_merge.sh`: example verification and merge entry script.

## How to run

prepare your raw multi-page VQA data in a `.jsonl` format. Each line should represent a document sample following the schema below:

```bash
{
  "image": ["path/to/page_0.jpg", "path/to/page_1.jpg", "..."],   # Paths to the multi-pages document page images.
  "gt_page": [1],                                                 # Indices of the ground-truth evidence pages (use [-1] if no evidence page).
  "conversations": [                                              # The raw Q-A pair.
    {
      "from": "human",
      "value": "<Question>"                                       # The specific question.
    },
    {
      "from": "gpt",
      "value": "<Answer>"                                         # The raw short answer to be distilled.
    }
  ]
}
```

The preprocessed MPDocVQA and DUDE datasets can be downloaded from [Link](https://pan.baidu.com/s/1_elbeGo2JJhOALsTVfCuhw?pwd=mq9m)

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

Secondary verification and merge the generated results:

```bash
cd ALR_data_generate
bash verify_and_merge.sh
```

The final distilled ALR training data can be downloaded from [Link](https://pan.baidu.com/s/19r9JdHCryP0LF15n1fO4Pg？pwd=mq9m)

