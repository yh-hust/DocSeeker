import json
import requests
import time
import os
import re
import io
import json
from PIL import Image
from openai import OpenAI
from glob import glob
from tqdm import tqdm
import numpy as np
import cv2
import re
from tqdm import tqdm
import multiprocessing
import argparse
import traceback
import base64

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")

client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def append_to_json(file_path, new_item):
    try:
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, list):
                print(f"Error: The root object of the JSON file '{file_path}' is not a list.")
                data = [] 
        else:
            data = []
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading or parsing file: {e}. Starting with an empty list.")
        data = []

    data.append(new_item)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def extract_answer(question, images, prompt, model_name="gpt-4o", image_base_path=""): 
    try:

        user_content = [
            {
                "type": "text",
                "text": question
            }
        ]
        for img in images:
            img_path = os.path.join(image_base_path, img)
            base64_image = encode_image(img_path)
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        response = client.chat.completions.create(
            model=model_name,
            max_tokens=1024,
            temperature=0.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            messages=[
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ]
        )
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"API Error: {e}")
        traceback.print_exc()
        return "Failed"

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                data.append(item)
    return data
def save_list_to_jsonl(data_list, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data_list:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def map_function(main_args):
    if multiprocessing.current_process().name == 'MainProcess':
        rank = 0
    else:
        rank = multiprocessing.current_process().name.split('-')[-1]
    model_name, prompt, item,args = main_args
    cur_try = 0
    max_try = args.max_try
    image_base_path = args.image_base_path
    temp_file = os.path.join(args.save_path,f'data_{rank}.json')
    question = item['conversations'][0]['value']
    page_id = [page_item+1 for page_item in item['gt_page']]
    images = [item['image'][page_item] for page_item in item['gt_page']]
    formatted_question = (
        f"Question: {question}\n"
        f"Document Page Number: {str(page_id)}\n"
        f"Document Page Content: <image>"
    )
    tags = ['<think>', '</think>', '<answer>', '</answer>']
    while cur_try<max_try:
        try:
            out = extract_answer(formatted_question,images,prompt,model_name,image_base_path)
            if not all(tag in out for tag in tags):
                cur_try+=1
                if cur_try>max_try:
                    item["expand_answer"]=out
                    append_to_json(temp_file,item)
                    return item
                continue
            item["expand_answer"] = out
            append_to_json(temp_file,item)
            return item
        except Exception as e:
            cur_try += 1
    item["expand_answer"] = 'Fails'
    append_to_json(temp_file,item)
    return item

def remove_duplicates(data):
    unique_data = {}
    for item in data:
        unique_data[item['id']] = item 
    return list(unique_data.values())

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_path", type=str, nargs='+')
    parser.add_argument("--save_path",type=str,default="output")
    parser.add_argument("--model_name", type=str, default="gemini-2.5-flash")
    parser.add_argument("--image_base_path",type=str,default="data/raw_data/images")
    parser.add_argument("--max_workers",type=int,default=10)
    parser.add_argument("--sys_prompt",type=str,default="./sys_prompt/prompt_for_answer_expand.md")
    parser.add_argument("--bg", type=int, default=0)
    parser.add_argument("--ed", type=int, default=100)
    parser.add_argument("--max_try",type=int,default=30)
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    data_list = []
    for file_path in args.original_path:
        data_list += read_jsonl(file_path)
    for idx,item in enumerate(data_list):
        item['id'] = idx

    temp_all_data = []
    if os.path.exists(os.path.join(args.save_path,'data.json')):
        temp_all_data += json.load(open(os.path.join(args.save_path,'data.json'),encoding='utf-8'))

    for i in range(args.max_workers+1):
        temp_file = os.path.join(args.save_path,f'data_{i}.json')
        
        if os.path.exists(temp_file):
            temp_all_data += json.load(open(temp_file,encoding='utf-8'))
    temp_all_data = sorted(temp_all_data,key = lambda item:item['id'])
    
    all_infered_datas = temp_all_data
    all_infered_datas = remove_duplicates(all_infered_datas)
    infered_ids = [d['id'] for d in all_infered_datas]

    with open("./sys_prompt/prompt_for_answer_expand.md","r", encoding="utf-8") as f:
        prompt = f.read()
    
    temp_data_list = []
    without_gt_num = 0
    for item in data_list:
        if item['gt_page'] == [-1]:
            without_gt_num += 1
            item['gt_page'] = [idx for idx in range(len(item['image']))]
            if len(item['gt_page'])>10:
                continue
        else:
            if len(item['image'])<5:
                continue
        temp_data_list.append(item)

    print(f"Total samples: {len(data_list)}")
    print(f"Found {without_gt_num} samples without evidence pages. Filtered out {len(data_list) - len(temp_data_list)} samples.")
    data_list = temp_data_list[args.bg:args.ed]
    print(f"Total valid samples: {len(temp_data_list)}. Generating from index {args.bg} to {args.ed}.")

    data_list = [d for d in data_list if d['id'] not in infered_ids]
    dist_args = [(args.model_name, prompt, item, args) for i, item in enumerate(data_list)]

    with multiprocessing.Pool(
        processes=args.max_workers
    ) as pool:
        new_datas = pool.map(map_function, dist_args)
        new_datas = all_infered_datas + new_datas
        new_datas = [d for d in new_datas if d!=None]

    new_datas = sorted(new_datas, key=lambda x: x["id"])
    with open(os.path.join(args.save_path,'data.json'), 'w', encoding='utf-8') as f:
        json.dump(new_datas, f, ensure_ascii=False, indent=4)
    