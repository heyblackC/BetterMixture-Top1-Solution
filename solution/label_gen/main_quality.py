import json
from transformers import AutoTokenizer
from tqdm import tqdm

import torch
import math
import json
import difflib
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.special import softmax

import os
import argparse

import datasets
import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

import multiprocessing
import os
from multiprocessing import Process, Manager


input_filename = "../../output/sft_data/mixture_raw_reweight.jsonl"

input_texts = []
with open(input_filename, 'r', encoding='utf-8') as jsonl_file:
    for line in jsonl_file:
        entry = json.loads(line)
        input_texts.append(entry)

# input_texts = input_texts[:16]
# input_texts = ["lorem ipsum", "Happy Birthday!", "Bienvenue", "这是一句中文话", "这是一句中文话qweqweqw"]


def _compute(
    predictions, model_id, batch_size: int = 16, add_start_token: bool = True, device = None, max_length = None
):

    # if device is not None:
    #     assert device in ["gpu", "cpu", "cuda"], "device should be either gpu or cpu."
    #     if device == "gpu":
    #         device = "cuda"
    # else:
    #     device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "hkust-nlp/deita-quality-scorer"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    # model = model.to(device)

    ret_list = []
    for entry in tqdm(predictions):
        input_text = entry["instruction"] + "\n" + entry["input"]
        resp_text = entry["output"]
        
        def infer_Quality(model, tokenizer, input_text, resp_text):
            quality_template = ("You are a helpful assistant. Please identify the quality score of the Response corresponding to the Question. \n #Question#:\n{instruction}\n#Response#:\n{output} \n##Quality: ")
            user_input = quality_template.format(instruction=input_text, output=resp_text)
            # input_ids = tokenizer.encode(user_input, return_tensors="pt")
            input_ids = tokenizer.encode(user_input, return_tensors="pt", truncation=True, max_length = 510)
            max_length = 512
            outputs = model.generate(input_ids, max_length=512, num_return_sequences=1, return_dict_in_generate=True, output_scores=True)
            logprobs_list = outputs.scores[0][0]
            score_logits = []
            id2score = {
                29896: "1",
                29906: "2",
                29941: "3",
                29946: "4",
                29945: "5",
                29953: "6"
            }
            score_template = np.array([1,2,3,4,5,6])
            for k in id2score:
                score_logits.append(logprobs_list[k])
            score_logits = np.array(score_logits)
            score_npy = softmax(score_logits, axis=0)
            score_npy = score_npy * score_template

            score_npy = np.sum(score_npy, axis=0)
            return score_npy

        score_npy = infer_Quality(model, tokenizer, input_text, resp_text)
        ret_list.append(score_npy)
    return ret_list

print(len(input_texts))
# 定义一个要在进程中运行的函数
def worker(begin, end, model_id, batch_size: int = 16, add_start_token: bool = False, device = None, max_length = None, shared_list = None):
    """线程调用的工作函数"""
    ret_pers = _compute(input_texts[begin:end+1], model_id, batch_size, add_start_token, device, max_length)
    
    # print(ret_pers)
    
    for idx in range(begin, end+1):
        shared_list[idx] = ret_pers[idx-begin]

# 创建并启动多个进程
def main():
    # 创建进程池

    manager = multiprocessing.Manager()
    shared_list = manager.list([-1]*len(input_texts))
    
    gpu_nums = 1

    processes = []

    model_path = '/mnt/workspace/peipao/jichunengli/Qwen-14B-Chat'

    texts_batch_size = len(input_texts) // gpu_nums

    for i in range(gpu_nums):  # 创建5个进程
        print(i)
        begin = i*texts_batch_size
        end = (i+1)*texts_batch_size-1
        
        p = multiprocessing.Process(target=worker, args=(begin, end, model_path, 2, False, f"cuda:{i}", 2048, shared_list))
        processes.append(p)
        p.start()  # 启动进程

    # 等待所有进程完成
    for p in processes:
        p.join()
    
    # print(f'修改后的列表: {shared_list}')
    
    output_dir = "output_quality_score_0201_reweight"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file_name = os.path.basename(input_filename).replace(".jsonl", ".json")
    with open(os.path.join(output_dir, output_file_name), "w") as w_file:
        json.dump(list(shared_list), w_file, ensure_ascii=False)

# 入口函数判断，确保是直接运行而非作为模块导入时执行
if __name__ == "__main__":
    main()
