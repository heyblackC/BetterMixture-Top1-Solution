# =============================================================================
# 以下是一个采用随机采样的示例，参赛者可自由编写文件内容。
# =============================================================================

import json
import random
from pathlib import Path
import numpy as np
import os

from tools.tool_inputcheck import input_check

# 设置随机种子，保证结果可复现
seed = 42
random.seed(seed)

# 设置随机种子，以确保随机操作的可重复性
np.random.seed(42)

# 如果使用 NumPy
# import numpy as np
# np.random.seed(seed)

# 如果使用 PyTorch
# import torch
# torch.manual_seed(seed)

# 开发套件根目录
base_dir = Path(__file__).resolve().parent.parent

# 输入输出路径
input_dir = base_dir / "input"
ratio_path = base_dir / "output" / "sft_data" / "ratio.json"
mixture_path = base_dir / "output" / "sft_data" / "mixture_raw_reweight.jsonl"

mixture_math_score_path = base_dir / "output" / "sft_data" / "mixture_raw_math_only.jsonl"

# 函数仅为示意，可自由改写
def generate_mixture(input_dir, ratio_path, mixture_path):
    
    # 依赖文件file_ratio.json
    total_count = 500000

    # 获取所有 jsonl 文件的路径列表
    # file_list = list(input_dir.glob("*.jsonl"))
    
    with open("file_ratio.json", 'r', encoding="utf-8") as file:
        # 使用json.load()将JSON内容解析为Python字典
        # 取了1w，如果连1000的好数据都没有，直接删除
        data_ratio_dict = json.load(file)
        for key,value in data_ratio_dict.items():
            if value < 1000:
                data_ratio_dict[key] = 0
        sumup = sum(data_ratio_dict.values())
        new_data_ratio_dict = {}
        for key,value in data_ratio_dict.items():
            new_data_ratio_dict[str(key).split(r"/")[-1].strip().replace(".json","")] = value/sumup
        # 加起来等于1
        data_ratio_dict = new_data_ratio_dict
        print(data_ratio_dict)
    
    file_list = [
        "../input/gpt4all.jsonl",
        "../input/Vicuna.jsonl",
        "../input/instruct.jsonl",
        "../input/HC3_ChatGPT.jsonl",
        "../input/alpaca_data.jsonl",
        "../input/sharegpt.jsonl",
        "../input/GPTeacher.jsonl",
        # "../input/COIG_translate_en.jsonl",
        "../input/dolly.jsonl",
        "../input/alpaca_gpt4_data.jsonl",
        # "../input/COIG_translate_zh.jsonl",
        "../input/finance_en.jsonl",
        "../input/belle_data0.5M_cn.jsonl",
        "../input/HC3_Human.jsonl",
        "../input/HC3_Chinese_ChatGPT.jsonl",
        "../input/sharegpt_zh.jsonl",
        "../input/instinwild_en.jsonl",
        "../input/instinwild_ch.jsonl",
        "../input/alpaca_gpt4_data_zh.jsonl",
        "../input/HC3_Chinese_Human.jsonl",
    ]
    
    # 按照采样概率抽取样本并写入到 mixture.jsonl
    with open(mixture_path, "w", encoding="utf-8") as mixture_file:
        for file_path in file_list:
            file_name = file_path.split(r"/")[-1].strip().replace(".jsonl","")
            with open(file_path, "r") as f:
                samples = f.readlines()
                np.random.shuffle(samples)
                samples = [json.loads(line) for line in samples]
                print("正则清理数据前，条数：", len(samples))
                samples = input_check(samples)
                samples = [item for item in samples if (not item["html_block"]) and (not item["code_detect"])]
                prob = data_ratio_dict[file_name]
                num_samples = int(prob * total_count)
                print(file_name)
                print(num_samples)
                selected_samples = samples[:num_samples]
                for line in selected_samples:
                    mixture_file.write(json.dumps(line, ensure_ascii=False) + '\n')

def list_in_str(target_list, text):
    for target in target_list:
        if target in text:
            return True
    return False

def generate_ratio(input_dir, ratio_path, mixture_path):
    # 依赖文件file_ratio.json
    with open("file_ratio.json", 'r', encoding="utf-8") as file:
        # 使用json.load()将JSON内容解析为Python字典
        # 取了1w，如果连1000的好数据都没有，直接删除
        data_ratio_dict = json.load(file)
        for key,value in data_ratio_dict.items():
            if value < 1000:
                data_ratio_dict[key] = 0
        sumup = sum(data_ratio_dict.values())
        new_data_ratio_dict = {}
        for key,value in data_ratio_dict.items():
            new_data_ratio_dict[str(key).split(r"/")[-1].strip().replace(".json",".jsonl")] = value/sumup
        # 加起来等于1
        data_ratio_dict = new_data_ratio_dict
        print(data_ratio_dict)
    
    with open(ratio_path, "w") as ratio_file:
        json.dump(data_ratio_dict, ratio_file, indent=4)

def insert_new_param():
    import json
    import pandas as pd
    import os
    
    # 读取json文件
    # deberta reward model score
    with open('./socre_deberta_0130_list.jsonl', "r", encoding="utf-8") as file:
        score_file = file.readlines()
        score_lines = [float(line.strip().strip("[").strip("]")) for line in score_file]

    with open('./qwen_1p8_instag.jsonl', "r", encoding="utf-8") as file:
        tag_file_list = file.readlines()

    with open("./mixture_raw_reweight_task_type.json", "r",  encoding="utf-8") as file:
        deita_scores = json.load(file)
    
    with open("./task_type_bge.jsonl", "r",  encoding="utf-8") as file:
        task_type_dicts = file.readlines()
        task_type_dicts = [json.loads(line) for line in task_type_dicts]
    
    with open("./socre_rerank_0201_list.jsonl", 'r', encoding="utf-8") as file:
        rank_score_lines = file.readlines()
        rank_score_lines = [float(score) for score in rank_score_lines]

    file1_path = '../output/sft_data/mixture_raw_reweight.jsonl'
    output_path = '../output/sft_data/mixture_raw_reweight_task_type_add_new_tag_score_reproduce.jsonl'

    assert len(score_lines) == len(tag_file_list) == len(deita_scores) == len(task_type_dicts) == len(rank_score_lines)
    print(len(score_lines))
    
    print_cnt = 200
    with open(file1_path, 'r') as file1 , open(output_path, 'w', encoding='utf-8') as outfile:
        lines_ = file1.readlines()
        print(len(lines_))

        for idx, line in enumerate(lines_):
            cur_data = json.loads(line)

            cur_data["instag"] = tag_file_list[idx]
            score = score_lines[idx]
            cur_data["reward"] = score
            cur_data["deita_quality"] = deita_scores[idx]
            
            cur_data["topic_label"] = task_type_dicts[idx]["topic_label"]
            cur_data["topic_prob"] = task_type_dicts[idx]["topic_prob"]
            
            cur_data["rerank"] = rank_score_lines[idx]


            outfile.write(json.dumps(cur_data, ensure_ascii=False) + '\n')

def filter_and_select_data_lines():
    import json
    import os
    # 假设我们有两个JSONL文件，file1.jsonl 和 file2.jsonl
    file1_path = '../output/sft_data/mixture_raw_reweight_task_type_add_new_tag_score_reproduce.jsonl'

    output_path = '../output/sft_data/mixture_raw_reweight_task_type_add_new_filtered_selected.jsonl'
    import numpy as np
    np.random.seed(42)
    from tqdm import tqdm
    import re

    from transformers import AutoTokenizer

    model_path = "baichuan-inc/Baichuan2-7B-Base"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

    total_tokens = 3.0e7

    # 打开两个文件以及用于输出的新文件
    # open('output.jsonl', 'w', encoding='utf-8')
    cnt = 0
    com_cnt = 0
    print_cnt = 100

    latex_pattern = re.compile(
        r'\\[a-zA-Z]+{.*?}|'             # 匹配LaTeX命令
        r'(\$\$?^[\<\d].+?\$\$?)',re.IGNORECASE
    )

    latext_cnt = 0
    seven_ask_cnt = 0
    multi_round_cnt = 0

    math_high_score_cnt = 0
    hallucination_cnt = 0
    rerank_cnt = 0
    openai_cnt = 0

    cache_list = []


    AI_language_out_cnt = 0
    sorry_cnt = 0
    code_detect_cnt = 0
    # 8000
    bins_cnt = 6000//100
    distribute_count_list = [20000]*bins_cnt
    print(distribute_count_list)
    story_create_cnt = 0
    with open(file1_path, 'r') as file1, open(output_path, 'w', encoding='utf-8') as outfile:
        # 使用zip_longest从两个文件中交替读取行，直到两个文件都读完
        # 如果其中一个文件比另一个长，它将继续读取，另一个文件的位置用None填充
        lines_ = file1.readlines()
        lines_new = []
        # prompt_toxi_detect = []
        print(len(lines_))
        for line in lines_:
            line = json.loads(line)
            lines_new.append(line)

        lines_ = lines_new
        lines_ = sorted(lines_, key=lambda x: x['deita_quality'], reverse=True)

        np.random.shuffle(lines_)


        for idx, cur_data in enumerate(tqdm(lines_)):
            # 写入来自第一个文件的行
            # cur_data = json.loads(line1)
            if len(cur_data["output"]) > 1:

                if cur_data["deita_quality"]< 1.5:
                    continue

                if cur_data["code_detect"]:
                    code_detect_cnt += 1
                    continue
                if cur_data["html_block"]:
                    continue
                unwanted_instag_words =["descriptive writing", "scene description",  "set description", "character development", "plot development", "imagery", "imagination", "fantasy", "fiction write", "storytelling", "role-playing", "role play"]

                if any(item in cur_data["instag"] for item in unwanted_instag_words):
                    story_create_cnt += 1
                    continue
                if cur_data["topic_prob"][0] > 0.40 and any(item in cur_data["topic_label"][:1] for item in ["编故事", "角色扮演", "故事续写"]):
                    story_create_cnt += 1
                    continue

                # if contains_unwanted_words(cur_data["output"]):
                #     AI_language_out_cnt += 1
                #     continue

                delete_keywords = ["语言模型", "抱歉", "我无法", "Sorry", "sorry", "apologize", "我道歉", "我错误地"]
                if any(keyword in cur_data["output"][:25] for keyword in delete_keywords):
                    sorry_cnt += 1
                    continue

                def contains_chinese(text):
                    chinese_pattern = re.compile(r'[\u4e00-\u9fa5]')
                    return bool(chinese_pattern.search(text))

                if not contains_chinese(cur_data["text"]) and cur_data["reward"] <= 0:
                    continue

                if cur_data["rerank"] < 0.30:
                    rerank_cnt += 1
                    continue

                if "openai" in cur_data["output"] or "chatgpt" in cur_data["output"].lower():
                    openai_cnt += 1
                    continue


                source_label = cur_data["meta"]["original_path"]

                lang_label = cur_data["meta"]["Lang"]


                if not contains_chinese(cur_data["text"]) and len(cur_data["text"].split(" ")) <= 10:
                    continue

                input_prompt = cur_data["instruction"] + "\n" + cur_data["input"]
                if not contains_chinese(input_prompt) and len(input_prompt.split(" ")) < 15:
                    continue

                if contains_chinese(input_prompt) and len(input_prompt) < 40:
                    continue

                if cur_data["o_i_ratio"] >= 20:
                    continue

                if len(cur_data["output"]) > 3000 or len(cur_data["output"]) < 5:
                    continue

                if len(cur_data["instruction"]+cur_data["input"]) > 6000:
                    continue

                bin_index =  len(cur_data["instruction"]+cur_data["input"]) // 100


                distribute_count_list[bin_index] -= 1

                if cur_data["math_score"] >= 0.80:
                    math_high_score_cnt += 1

                if cur_data["seven_ask"]:
                    seven_ask_cnt += 1


                ###### new below #####
                tokens = tokenizer.encode(cur_data["text"])
                if len(tokens) >= 4080:
                    continue
                total_tokens -= len(tokens)
                if total_tokens <= 0:
                    print("获取够了，跳出！！！")
                    break
                cache_list.append(cur_data)
                cnt += 1
        print("cache_list",len(cache_list))
        # np.random.shuffle(cache_list)
        for cur_data in tqdm(cache_list):
            # del cur_data["__dj__stats__"]
            outfile.write(json.dumps(cur_data, ensure_ascii=False) + '\n')

    print("distribute_count_list",distribute_count_list)
    print("latext_cnt",latext_cnt)
    print(f"Interleaved JSONL file created at {output_path}")
    print(cnt)
    print(com_cnt)
    print("math_high_score_cnt",math_high_score_cnt)
    print("seven_ask_cnt",seven_ask_cnt)
    print("multi_round_cnt",multi_round_cnt)
    print("sorry_cnt", sorry_cnt)
    print("AI_language_out_cnt", AI_language_out_cnt)
    print("code_detect_cnt",code_detect_cnt)
    print("openai_cnt",openai_cnt)
    print("rerank_cnt",rerank_cnt)
    print("hallucination_cnt",hallucination_cnt)
    print("story_create_cnt",story_create_cnt)
    # with open(file1_path, 'r') as file1:
    #     lines = file1.readlines()
    #     print(len(lines))

def data_juicer_filtered():
    import os
    os.system('dj-process --config alpaca-cot-zh-refine_0205_reweight_for_new_tag_use_keep_cn.yaml')

def knn_semantic_data_filtered():
    import logging
    import json as json
    from collections import namedtuple
    import sys
    import os
    from tqdm import tqdm
    import torch
    import transformers
    from pynndescent import NNDescent
    from sentence_transformers import SentenceTransformer
    import pandas as pd
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    logger = logging.getLogger(__file__)

    print_cnt = 1000
    sent_model = SentenceTransformer("maidalun1020/bce-embedding-base_v1")
    print(sent_model)
    sent_model.to(torch.device("cuda:0"))
    print(sent_model.device)
    
    file1_path = "../output/sft_data/juicer_out/task_type_mixback_new_keep_cn.jsonl"
    output_file = "../output/sft_data/juicer_out/task_type_mixback_new_keep_cn_knn_filtered.jsonl"

    with open(file1_path, 'r') as file1:
        lines_ = file1.readlines()
        raw_data = [json.loads(line) for line in lines_]
        raw_data = sorted(raw_data, key=lambda x: x['deita_quality'], reverse=True)
        
        db_dataset = [line["text"] for line in raw_data]


    db_embeddings = sent_model.encode(db_dataset, show_progress_bar=True)
    print("start nndesent", len(db_embeddings))

    anns_index = NNDescent(db_embeddings)
    ann_distance = anns_index.neighbor_graph[1]
    n_neighbor_index = anns_index.neighbor_graph[0]
    
    print(n_neighbor_index.shape)
    print("NNDesent done")    

    #ann_index, ann_distance = anns_index.query(db_embeddings, k=6)
    #print("query done")
    delete_cnt = 0
    save_data = []
    selected_book = [0] * len(raw_data)
    print(len(selected_book))
    # threshold = 0.55
    # 原先的最好用的是0.60的threshold
    threshold = 0.61
    print(ann_distance[0])
    for i, one_data in enumerate(tqdm(raw_data, desc="save data")):
        f_nearest_dis = 1e9
        index_neighbor = None
        for j in range(1, n_neighbor_index.shape[1]):
            # if ann_distance[i,j] > threshold:
            #     break
            index_neighbor = n_neighbor_index[i,j]
            # 获取距离最近的点的实际下标，然后判断它是否已经访问过
            if selected_book[index_neighbor] == 1:
                f_nearest_dis = ann_distance[i,j]
                break
        if f_nearest_dis < threshold:
            delete_cnt += 1
            if print_cnt > 0:
                print_cnt -= 1
                print("begin===================")
                print(raw_data[index_neighbor])
                print(raw_data[i])
                print("end===================")
            continue
        # elif f_nearest_dis == 1e9:
        #     f_nearest_dis = ann_distance[]
        selected_book[i] = 1
        
        one_data["knn_n"] = round(float(f_nearest_dis),2)
        save_data.append(one_data)
    
    with open(output_file, "w", encoding="utf-8") as fd:
        for one_data in save_data:
            fd.write(json.dumps(one_data, ensure_ascii=False) + '\n')
    print("delete_cnt",delete_cnt)
    print(len(save_data))

def extract_math_using_math_score(input_dir, mixture_math_score_path):
    
    # mixture_math_score_path

    file_list = [
        "../input/gpt4all.jsonl",
        "../input/Vicuna.jsonl",
        "../input/instruct.jsonl",
        "../input/HC3_ChatGPT.jsonl",
        "../input/alpaca_data.jsonl",
        "../input/sharegpt.jsonl",
        "../input/GPTeacher.jsonl",
        # "../input/COIG_translate_en.jsonl",
        "../input/dolly.jsonl",
        "../input/alpaca_gpt4_data.jsonl",
        # "../input/COIG_translate_zh.jsonl",
        "../input/finance_en.jsonl",
        "../input/belle_data0.5M_cn.jsonl",
        "../input/HC3_Human.jsonl",
        "../input/HC3_Chinese_ChatGPT.jsonl",
        "../input/sharegpt_zh.jsonl",
        "../input/instinwild_en.jsonl",
        "../input/instinwild_ch.jsonl",
        "../input/alpaca_gpt4_data_zh.jsonl",
        "../input/HC3_Chinese_Human.jsonl",
    ]
    
    # 按照采样概率抽取样本并写入到 mixture.jsonl
    with open(mixture_math_score_path, "w", encoding="utf-8") as mixture_file:
        for file_path in file_list:
            file_name = file_path.split(r"/")[-1].strip().replace(".jsonl","")
            with open(file_path, "r") as f:
                samples = f.readlines()
                np.random.shuffle(samples)
                samples = [json.loads(line) for line in samples]
                print("正则清理数据前，条数：", len(samples))
                samples = input_check(samples)
                samples = [item for item in samples if (not item["html_block"]) and (not item["code_detect"]) and (item["math_score"] > 0.70)]
                for line in samples:
                    mixture_file.write(json.dumps(line, ensure_ascii=False) + '\n')

def extract_regex_math():
    from get_mixture_0128_extract_math import generate_mixture, validate_math_expression
    mixture_regex_path = base_dir / "output" / "sft_data" / "mixture_raw_math_only_regex.jsonl"
    generate_mixture(input_dir, ratio_path, mixture_regex_path)
    validate_math_expression()
    
def data_juicer_filtered_math_score():
    import os
    os.system('dj-process --config alpaca-cot-zh-refine_0128_reweight_math.yaml')
    
def insert_new_param_for_math_score():
    import json
    import pandas as pd
    import os
    
    with open("./task_type_bge_math_score.jsonl", "r",  encoding="utf-8") as file:
        task_type_dicts = file.readlines()
        task_type_dicts = [json.loads(line) for line in task_type_dicts]

    file1_path = '../output/sft_data/math_scorer_juicer/mixture_raw_math_only_juicer.jsonl'
    output_path = '../output/sft_data/math_scorer_juicer/mixture_raw_math_only_juicer_task_type.jsonl'
    
    print_cnt = 200
    with open(file1_path, 'r') as file1 , open(output_path, 'w', encoding='utf-8') as outfile:
        lines_ = file1.readlines()
        print(len(lines_))
        assert len(lines_) == len(task_type_dicts)

        for idx, line in enumerate(lines_):
            cur_data = json.loads(line)

            cur_data["topic_label"] = task_type_dicts[idx]["topic_label"]
            cur_data["topic_prob"] = task_type_dicts[idx]["topic_prob"]

            outfile.write(json.dumps(cur_data, ensure_ascii=False) + '\n')

# 第一步：混合原始数据
generate_mixture(input_dir, ratio_path, mixture_path)

# 第二步：输出混合ratio_dict
generate_ratio(input_dir, ratio_path, mixture_path)

# 第三步：将一些新字段加入到数据中
insert_new_param()

# 第四步：利用新字段和指令长度等条件进行过滤,需要访问huggingface的baichuan-inc/Baichuan2-7B-Base模型
filter_and_select_data_lines()

# 第五步：利用data-juicer进行过滤,需要依赖data-juicer工具,需要使用simhash,minhash等功能
data_juicer_filtered()

# 第六步：knn语义相似度数据过滤
knn_semantic_data_filtered()

# 第七步：获取math_high_score数据和用正则获取纯数学数据
# 获取数据并进行data_juicer去重
extract_math_using_math_score(input_dir, mixture_math_score_path)
data_juicer_filtered_math_score()
insert_new_param_for_math_score()

# 第八步：获取正则数学数据
extract_regex_math()

# 最后一步：混合所有的数据
os.system('python mix_back_final_solution.py')
