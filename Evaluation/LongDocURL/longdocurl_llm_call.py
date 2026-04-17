import traceback
import multiprocessing
import json
import os
import re
import requests
from openai import OpenAI

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")

client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )

system_prompt = "You are an expert in visual document question-answering, please answer our questions based on the given images.\n"

def append_to_json(file_path, new_data):
    # 读取已有数据（如果存在），并追加新数据
    # 将合并后的数据保存回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

def collect_stream_data(stream_generator):
    """Collect reasoning and content from SSE stream generator"""
    reasoning_parts, content_parts = [], []

    for line in stream_generator:
        # 生成器已经返回了包含换行符的行，需要去除空白
        line = line.strip()

        if not line or not line.startswith('data: '):
            continue

        data = line[6:].strip()
        if data == '[DONE]':
            continue

        try:
            parsed = json.loads(data)

            # 安全检查 choices 数组
            if not parsed.get('choices') or len(parsed['choices']) == 0:
                continue

            delta = parsed['choices'][0].get('delta', {})

            if reasoning := delta.get('reasoning_content'):
                reasoning_parts.append(reasoning)

            if content := delta.get('content'):
                content_parts.append(content)
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            # 调试：打印解析失败的数据
            print(f"解析失败: {e}, 数据: {data[:100]}")
            continue

    return ''.join(reasoning_parts), ''.join(content_parts)

def get_msg_format(prompt):
    content = [{"type": "text", "text": prompt}]
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    return messages

def call_llm(prompt, model_name, temperature=0.1, seed=42, max_tokens=4096):
    msgs = get_msg_format(prompt)
    try:
        # 4. 调用 API
        # 参照 SmolDocling 论文实验，评估时通常设定较低的温度系数以保证结果稳定 [cite: 189, 235]
        completion = client.chat.completions.create(
            model=model_name,
            messages=msgs,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,  # 增加 seed 参数以提高实验可复现性
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # 5. 解析并返回内容
        response = completion.choices[0].message.content
        return response

    except Exception as e:
        print(f"调用 LLM 时发生异常: {e}")
        # 获取详细的错误堆栈
        traceback.print_exc()
        return "Failed"



def map_function(args):
    
    model_name, item, prompt = args
    #import pdb;pdb.set_trace()
    #item['pred'] = [item['pred']]
    if isinstance(item["pred"],list):
        item["pred"] = ' '.join(item["pred"])
    while True:
        try:
            #import pdb;pdb.set_trace()
            prompt = prompt + "\nQuestion: " + item['question'] + "\nAnalysis: " + item["pred"]
            extractor_result = call_llm(prompt, model_name)
            #item["extracted_res"] = extractor_result
            try:
                import re
                concise_answer = re.findall(r"<concise_answer>(.*?)</concise_answer>", extractor_result, re.DOTALL)[0]
                answer_format = re.findall(r"<answer_format>(.*?)</answer_format>", extractor_result, re.DOTALL)[0]
            except:
                concise_answer = "Fail to extract"
                answer_format = "None"
            
            try:
                item["extracted_res"] = eval(concise_answer) if not isinstance(eval(concise_answer), set) else list(eval(concise_answer))
            except:
                item["extracted_res"] = concise_answer
            
            print("=" * 60)
            print(f"问题: {item['question']}")
            print(f"原始预测 (用于分析): {item['pred']}")
            print(f"最终提取的答案: {item['extracted_res']}")
            print("=" * 60 + "\n")

            return item
            
        except Exception as e:
            print(e)
            traceback.print_exc()


def llm_process(data_list, model_name):
    #import pdb;pdb.set_trace()
    infered_datas = []
    with open("./LongDocURL/prompt_for_answer_extraction.md") as f:
        prompt = f.read()
    prompt = system_prompt + prompt
    dist_args = [(model_name, item, prompt) for i, item in enumerate(data_list)]
    with multiprocessing.Pool(
        processes=50
    ) as pool:
        new_datas = pool.map(map_function, dist_args)
        new_datas = infered_datas+new_datas
        new_datas = [d for d in new_datas if d!=None]

    return new_datas