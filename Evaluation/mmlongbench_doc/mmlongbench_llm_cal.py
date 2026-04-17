from PIL import Image
from openai import OpenAI
import json
import base64
import traceback
import requests
import multiprocessing
import os

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")

client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )

def extract_answer(question, output, prompt, model_name="gpt-4o"):
    try:
        # 构建消息序列
        # 参考论文中的任务定义，模型需要处理识别、定位和关联 subtasks [cite: 69, 70, 71]
        messages = [
            {
                "role": "user", 
                "content": prompt
            },
            {
                "role": "assistant",
                "content": f"\n\nQuestion:{question}\nAnalysis:{output}\n"
            }
        ]

        # 调用 API 进行推理
        # 注意：此处使用的温度系数为 0.0 以保证评估结果的一致性 [cite: 189]
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=512,
            temperature=0.0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # 获取返回内容
        response = completion.choices[0].message.content
        
    except Exception as e:
        print(f"提取答案时发生错误: {e}")
        traceback.print_exc()
        response = "Failed"
    
    return response

def append_to_json(file_path, new_data):
    # 读取已有数据（如果存在），并追加新数据
    # 将合并后的数据保存回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def map_function(args):
    model_name, item, prompt = args
    while True:
        try:
            #item["id"] = i
            question = item["question"]
            output = item["pred"][0]
            if output=='OOM':
                out = 'Extracted answer: Fail to answer\nAnswer format: String'
                item["extracted_res"]=out
                return item
            if output=='' or output==' ':
                out = 'Not answerable'
                item["extracted_res"]=out
                return item
            #text_input = "\n\nQuestion:{}\nAnalysis:{}\n".format(question, output)
            print(output)
            #import pdb;pdb.set_trace()
            out = extract_answer(question,output,prompt,model_name)
            assert out != "Failed"
            item["extracted_res"]=out
            print(f"item_id:{item['id']}\nLLM output:{out}")
            return item
        except Exception as e:
            print(e)
            traceback.print_exc()

def llm_process(data_list, model_name):
    infered_datas = []
    with open("./mmlongbench_doc/eval/prompt_for_answer_extraction.md") as f:
        prompt = f.read()
    prompt = prompt
    dist_args = [(model_name, item, prompt) for i, item in enumerate(data_list)]
    #import pdb;pdb.set_trace()
    #map_function(dist_args[0])
    #import pdb;pdb.set_trace()
    with multiprocessing.Pool(
        processes=50
    ) as pool:
        new_datas = pool.map(map_function, dist_args)
        new_datas = infered_datas+new_datas
        new_datas = [d for d in new_datas if d!=None]

    return new_datas