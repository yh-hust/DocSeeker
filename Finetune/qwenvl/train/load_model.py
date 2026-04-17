from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer
from transformers import AutoModelForCausalLM, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
import torch
from qwen_vl_utils import process_vision_info
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from qwenvl.MoBA_attn.moba import register_moba, MoBAConfig,register_moba_to_qwenvl2_5
from time import time

if __name__=='__main__':
    model_name_or_path='/cache/weight/Wukong-7B'
    attn_implementation = "moba"
    if "moba" in attn_implementation:
        register_moba_to_qwenvl2_5(MoBAConfig(4096, 12))
    model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
        ).cuda().eval()

    processor = AutoProcessor.from_pretrained(model_name_or_path)
    bg = time()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/home/ma-user/work/test.jpg"},
                {"type": "image", "image": "/home/ma-user/work/test.jpg"},
                {"type": "image", "image": "/home/ma-user/work/test.jpg"},
                {"type": "image", "image": "/home/ma-user/work/test.jpg"},
                {"type": "image", "image": "/home/ma-user/work/test.jpg"},
                {"type": "image", "image": "/home/ma-user/work/test.jpg"},
                {"type": "image", "image": "/home/ma-user/work/test.jpg"},
                {"type": "image", "image": "/home/ma-user/work/test.jpg"},
                {"type": "image", "image": "/home/ma-user/work/test.jpg"},
                {"type": "text", "text": "Read this document."},
            ],
        }
    ]
    text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ) #(2000*9,1120)
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    #import pdb;pdb.set_trace()
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print(output_text)
    ed = time()
    print(ed-bg)