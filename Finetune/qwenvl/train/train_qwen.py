# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwenvl.train.trainer
from trainer import replace_qwen2_vl_attention_class

from transformers import (
    AutoModelForCausalLM,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from qwenvl.data.data_qwen import make_supervised_data_module

from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer
import torch.distributed as dist

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def setup_distributed():
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",  # 或 'gloo'（CPU），'nccl' 推荐用于GPU
            init_method="env://",  # 通过环境变量初始化（torchrun 默认方式）
        )
        print(f"✅ Distributed init: {torch.distributed.is_initialized()}")
        print(f"🌐 World size: {torch.distributed.get_world_size() if torch.distributed.is_initialized() else 'N/A'}")
        print(f"🔢 Rank: {torch.distributed.get_rank() if torch.distributed.is_initialized() else 'N/A'}")

def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for idx, (n, p) in enumerate(model.model.named_parameters()):
            # if 'layers.' not in n:
            #     p.requires_grad = True
            #     continue
            # layer_num = int(n.split('.')[1])
            # if layer_num < 60:
            #     p.requires_grad = False
            #     continue
            # print(idx, n)
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False
    # exit()
global_tokenizer = None
def compute_loss(outputs, labels, num_items_in_batch):
    def get_mask(mask_str, shift_labels):
        tokens = global_tokenizer.tokenize(mask_str)
        num_ids = global_tokenizer.convert_tokens_to_ids(tokens)
        masks = (shift_labels >= num_ids[0])*(shift_labels <= num_ids[-1])
        return masks
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    
    #shift_logits = outputs.logits[..., :-1, :].view(-1, 152064)
    shift_logits = outputs.logits[..., :-1, :].reshape(-1, outputs.logits.shape[-1])
    shift_labels = labels[..., 1:].view(-1)
    
    num_masks = get_mask("0123456789", shift_labels)
    # choice_masks = get_mask("ABCD", shift_labels)


    weights = torch.ones_like(shift_labels)
    weights[num_masks] = 2.0
    # print(weights.shape)
    # print(weights.min(), weights.max())
    train_mask = shift_labels>-1
    loss = loss_fct(shift_logits, shift_labels)
    loss = torch.sum(loss*weights*train_mask)/torch.sum(weights*train_mask)
    # print(loss)
    return loss

def train(attn_implementation="flash_attention_2"):
    global local_rank
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    attn_implementation=model_args.attn_implementation

    if "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = Qwen2VLImageProcessor.from_pretrained(
            model_args.model_name_or_path,
        )
        data_args.model_type = "qwen2vl"

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    set_model(model_args, model)
    global global_tokenizer
    global_tokenizer = tokenizer

    if torch.distributed.get_rank() == 0:
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    #import pdb;pdb.set_trace()
    data_module['train_dataset'][0]
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args,compute_loss_func=compute_loss, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)
    if "wukong" in model_args.model_name_or_path.lower():
        source_path = os.path.join(model_args.model_name_or_path, "chat_template.json")
        template_path = os.path.join(training_args.output_dir, "chat_template.json")
        shutil.copy2(source_path, template_path)

        source_path = os.path.join(model_args.model_name_or_path, "modeling_wukong_chat.py")
        template_path = os.path.join(training_args.output_dir, "modeling_wukong_chat.py")
        shutil.copy2(source_path, template_path)


    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    setup_distributed()
    train(attn_implementation="flash_attention_2")
