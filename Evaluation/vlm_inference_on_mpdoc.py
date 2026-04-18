import os
import re
import math
import json
import argparse
import fitz
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm
import torch.distributed as dist
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import random
import numpy as np
import traceback
HAS_VAL  = ['dude','mpdocvqa','slidevqa']
DOCUMENT_PATH = {
    'dude':'dude/pdfs',
    'docbench':'DocBench/pdfs',
    'mmlongbench-doc':'mmlongbench-doc/documents'
}

def get_rank_and_world_size():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def save_inference_results(all_infered_datas, save_path):
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(all_infered_datas, f, ensure_ascii=False,indent=4)

def append_to_json(file_path, new_data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

def extract_images(sample, document_path, max_pages=120, resolution=144):
    image_list = list()
    doc_name = re.sub(r"\.pdf$", "", sample["doc_id"]).split("/")[-1]
    
    with fitz.open(os.path.join(document_path, sample["doc_id"])) as pdf:
        for index, page in enumerate(pdf[:max_pages]):
            if not os.path.exists(f"./{document_path.replace('/pdfs','')}/tmp/{doc_name}_{index}.png") and not os.path.exists(f"./{document_path.replace('/pdfs','')}/tmp/{doc_name}_{index}.jpg"):
                im = page.get_pixmap(dpi=resolution)
                if not os.path.exists(f"./{document_path.replace('/pdfs','')}/tmp/{doc_name}_{index}.png") and not os.path.exists(f"./{document_path.replace('/pdfs','')}/tmp/{doc_name}_{index}.jpg"):
                    try:
                        im.save(f"./{document_path.replace('/pdfs','')}/tmp/{doc_name}_{index}.png")
                    except:
                        pass
            if os.path.exists(f"./{document_path.replace('/pdfs','')}/tmp/{doc_name}_{index}.png"):
                image_list.append(f"./{document_path.replace('/pdfs','')}/tmp/{doc_name}_{index}.png")
            elif os.path.exists(f"./{document_path.replace('/pdfs','')}/tmp/{doc_name}_{index}.jpg"):
                image_list.append(f"./{document_path.replace('/pdfs','')}/tmp/{doc_name}_{index}.jpg")

    return image_list

def get_mpdoc_benchmark(args):
    if args.subset=='val':
        assert args.dataset_name.lower() in HAS_VAL
    data_list = []
    if args.dataset_name.lower() == 'dude':
        file_path = f"dude/sample_{args.subset}.json"
        data_list = json.load(open(file_path,'r',encoding='utf-8'))
        image_path = f"dude/tmp"
    elif args.dataset_name.lower() == 'mpdocvqa':
        file_path = f"mpdocvqa/sample_{args.subset}.json"
        data_list = json.load(open(file_path,'r',encoding='utf-8'))
        image_path = f"mpdocvqa/tmp"
    elif args.dataset_name.lower() == 'longdocurl':
        assert args.subset == 'test'
        file_path = f"LongDocURL/sample.json"
        data_list = json.load(open(file_path,'r',encoding='utf-8'))
        image_path = f"LongDocURL/tmp"
        for item in data_list:
            item['images'] = [img.split('.png')[0] for img in item['images']]
    elif args.dataset_name.lower() == 'mmlongbench_doc':
        assert args.subset == 'test'
        file_path = f"mmlongbench_doc/sample.json"
        image_path = f"mmlongbench_doc/tmp"
        data_list = json.load(open(file_path,'r',encoding='utf-8'))
    elif args.dataset_name.lower() == "slidevqa":
        file_path = f"SlideVQA/sample_{args.subset}.json"
        data_list = json.load(open(file_path,'r',encoding='utf-8'))
        image_path = f"SlideVQA/tmp"
        for item in data_list:
            item['images'] = [img.split('.png')[0] for img in item['images']]
        
    else:
        raise NotImplementedError
    
    for idx, item in enumerate(data_list):
        item['id'] = idx
        item['image_path']=image_path
    return data_list 

def qwen_chat(model,processor,question,images,args):
    if not args.use_direct:
        if args.use_page_named:
            img_list = []
            for idx,image_item in enumerate(images):
                img_list.append({'type': 'text', 'text': f'page {idx+1}:\n'})
                img_list.append({'type': 'image', 'image':image_item})
            
        else:
            #print("not use page_named")
            img_list = [{"type":"image","image":image} for image in images]
        messages = [
                    {
                        "role":"user",
                        "content": [{"type": "text", "text": question}] + img_list
                    }
                ]
    else:
        SYSTEM_PROMPT = "You are an expert in visual document question-answering. Please answer the question based on the given images. Provide only the short answer without any explanation. If the answer is a word or phrase, output just that word or phrase. If the information is insufficient to answer the question, you should answer with 'not-answerable'\n Question:"
        #print(f"评估inference-time scaling模型，需要完成SYSTEM_PROMPT")
        #print(f"SYSTEM_PROMPT:{SYSTEM_PROMPT}")
        if args.use_page_named:
            image_combine_list = []
            
            for idx,image_item in enumerate(images):
                image_combine_list.append({'type': 'text', 'text': f'page {idx+1}:\n'})
                image_combine_list.append({'type': 'image', 'image':image_item})
        else:
            image_combine_list = [{"type":"image","image":image} for image in images]
        messages = [
                    {
                        "role":"user",
                        "content": [{"type": "text", "text": SYSTEM_PROMPT+ '\n' + question}] + image_combine_list
                    }
                ]
    #import pdb;pdb.set_trace()
    with torch.no_grad():
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
        )
        inputs = inputs.to(model.device)
        #import pdb;pdb.set_trace()
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_tokens)
        generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
        response = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        print(f"rank:{rank}-pagenum:{len(images)}-query:{messages}-res:{response}")
        torch.cuda.empty_cache()
    return response


def inference(model,processor,dataset,save_name,args):
    infered_datas = []
    rank, world_size = get_rank_and_world_size()
    sheet_indices = list(range(rank, len(dataset), world_size))
    lt = len(sheet_indices)
    temp_path = os.path.join(args.save_dir,f'{save_name}_{rank}.json')
    if os.path.exists(temp_path):
        infered_datas = json.load(open(temp_path,'rb'))
    else:
        infered_datas = []
    processed_id = [item['id'] for item in infered_datas]
    # if rank == 0:
    #     import pdb;pdb.set_trace()
    # else:
    #     dist.barrier()
    #import pdb;pdb.set_trace()
    for i in tqdm(range(lt)):
        data = dataset[sheet_indices[i]]
        if data['id'] in processed_id:
            continue
        #import pdb;pdb.set_trace()
        if data.get('images') is None and DOCUMENT_PATH.get(args.dataset_name.lower()) is not None:
            img_list = extract_images(data,DOCUMENT_PATH[args.dataset_name.lower()],args.max_pages,args.resolution)
        else:
            img_list = []
            assert data.get('images') is not None
            img_list = [os.path.join(data['image_path'],f'{img}.jpg') if os.path.exists(os.path.join(data['image_path'],f'{img}.jpg')) else os.path.join(data['image_path'],f'{img}.png') for img in data['images'][:args.max_pages]]
        
        question,id = data['question'],data['id']
        answer = data['answer'] if data.get('answer',None) is not None else data['answers']
        
        try:
            #import pdb;pdb.set_trace()
            response = qwen_chat(model, processor, question, img_list, args)
        except Exception as e:
            print(f"[Error] 推理失败：{type(e).__name__}: {e}")
            traceback.print_exc()
            response = "OOM"
            torch.cuda.empty_cache()
        #import pdb;pdb.set_trace()
        sample = data.copy()
        sample['id'] = id
        sample['question'] = question
        sample['page_num'] = len(img_list)
        sample['answer'] = answer
        sample['pred'] = response
        # sample = {
        #     'id':id,
        #     'doc_id':data['doc_id'],
        #     'question':question,
        #     'page_num':len(img_list),
        #     'answer':answer,
        #     'pred':response,
        #     'evidence_sources':data.get('evidence_sources')
        # }
        # if args.dataset_name.lower()=='longdocurl':
        #     sample["answer_format"] = data["answer_format"]
        #     sample["evidence_pages"] = data["evidence_pages"]
        # if args.dataset_name.lower()=='docbench':
        #     sample["evidence_pages"] = data["evidence_pages"]
        infered_datas.append(sample)
        append_to_json(temp_path,infered_datas)

def load_model(args):

    if 'qwen' in args.model_name:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype="auto",attn_implementation=args.attn_implementation,trust_remote_code=True
            ).cuda().eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype="auto",attn_implementation=args.attn_implementation,trust_remote_code=True
            ).cuda().eval()
    if 'qwen' in args.model_name:
        max_pixels = 1024*28*28
        min_pixels = 512*28*28
        processor = AutoProcessor.from_pretrained(args.model_path, min_pixels=min_pixels, max_pixels=max_pixels,trust_remote_code=True)
    return model,processor

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Evaluation script for DocSeeker on multi-page document VQA datasets.")
    
    parser.add_argument("--dataset_name", type=str, default="mpdocvqa", 
                        choices=["dude", "mpdocvqa", "longdocurl", "slidevqa", "mmlongbench_doc"],
                        help="Name of the evaluation dataset.")
    parser.add_argument("--subset", type=str, default="val", choices=["val", "test"],
                        help="Dataset split to evaluate on (e.g., val or test).")
    parser.add_argument("--save_dir", type=str, default="output/qwen",
                        help="Directory to save the inference results and logs.")
    
    parser.add_argument("--model_name", type=str, default="qwen2_5_vl",
                        help="Identifier for the model architecture or configuration.")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Local path or Hugging Face repository ID of the model weights.")
    
    parser.add_argument("--max_pages", type=int, default=120,
                        help="Maximum number of document pages to process, useful for truncating ultra-long documents.")
    parser.add_argument("--resolution", type=int, default=144,
                        help="Image resolution setting for the vision encoder.")
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="Maximum number of tokens to generate during inference.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for text generation. Use lower values for more deterministic outputs.")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                        help="Attention mechanism implementation to use (e.g., flash_attention_2 for memory efficiency).")
    
    parser.add_argument("--use_direct", action="store_true",
                        help="If specified, the model generates direct answers without the structured ALR reasoning process.")
    parser.add_argument("--use_page_named", action="store_true",
                        help="If specified, prepends page identifiers (e.g., 'Page 1:') to the visual tokens of each page.")
    
    parser.add_argument("--rank", default=0, type=int,
                        help="Global rank for distributed evaluation.")
    parser.add_argument("--local_rank", default=0, type=int,
                        help="Local rank for distributed evaluation (handled automatically by deepspeed/torchrun).")
    
    args = parser.parse_args()

    dataset_name = args.dataset_name
    
    if dataset_name == 'dude':
        folder_path = 'dude'
    elif dataset_name == 'mpdocvqa':
        folder_path = 'mpdocvqa'
    elif dataset_name == 'longdocurl':
        folder_path = 'LongDocURL'
        assert args.subset == 'test', "LongDocURL only supports 'test' subset."
    elif dataset_name == 'slidevqa':
        folder_path = 'SlideVQA'
    elif dataset_name == 'mmlongbench_doc':
        folder_path = 'mmlongbench_doc'
        assert args.subset == 'test', "MMLongBench_doc only supports 'test' subset."
    
    save_dir = args.save_dir
    os.makedirs(save_dir,exist_ok=True)
    save_name = f'{args.dataset_name}'
    save_path = os.path.join(save_dir,f'{save_name}.json')
    if dist.is_available():
        dist.init_process_group(backend='nccl')

    rank,world_size = get_rank_and_world_size()
    if world_size>1:
        torch.cuda.set_device(rank)
    torch.cuda.synchronize()

    if rank==0:
        for arg_name, arg_value in vars(args).items():
            print(f"  {arg_name}: {arg_value}")
        temp_all_data = []
        if os.path.exists(save_path):
            temp_all_data += json.load(open(save_path,'r',encoding='utf-8'))
        for i in range(8):
            temp_file = os.path.join(save_dir,f'{save_name}_{i}.json')
            if os.path.exists(temp_file):
                with open(temp_file, 'rb') as f:
                    temp_all_data.extend(json.load(f))
                
        temp_all_data = sorted(temp_all_data,key = lambda item:item['id'])
        unique_data = {}
        for item in temp_all_data:
            if item['id'] not in unique_data:
                unique_data[item['id']] = item
    else:
        temp_all_data = None
    
    if world_size > 1:
        dist.barrier()
        temp_all_data_list = [temp_all_data]
        dist.broadcast_object_list(temp_all_data_list, src=0)
        temp_all_data = temp_all_data_list[0] 
    all_infered_datas = []
    #import pdb;pdb.set_trace()
    if os.path.exists(save_path):
        all_infered_datas += json.load(open(save_path,'r',encoding='utf-8'))
    infered_ids = [d['id'] for d in all_infered_datas+temp_all_data]

    if world_size > 1:
        if rank==0:
            dataset = get_mpdoc_benchmark(args)
        else:
            dataset = None
    else:
        dataset = get_mpdoc_benchmark(args)
    
    dist.barrier()
    
    if world_size > 1:
        dist.barrier()
        dataset_list = [dataset]
        dist.broadcast_object_list(dataset_list, src=0)
        dataset = dataset_list[0]
    dataset_len = len(dataset)
    dataset = [item for item in dataset if item['id'] not in infered_ids]
    model,processor = load_model(args)
    inference(model,processor,dataset,save_name,args)

    if world_size > 1:
        dist.barrier() 
    

    if rank == 0:
        for i in range(8):
            temp_file = os.path.join(args.save_dir,f'{save_name}_{i}.json')
            if os.path.exists(temp_file):
                with open(temp_file, 'rb') as f:
                    all_infered_datas.extend(json.load(f))
        unique_dict = {item['id']: item for item in all_infered_datas}
        data = list(unique_dict.values())

        all_infered_datas = sorted(data,key = lambda item:item['id'])
        
        assert len(all_infered_datas)==dataset_len
        save_inference_results(all_infered_datas,save_path)
