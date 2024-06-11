#!/usr/bin/env python
# coding: utf-8




import json
import pandas as pd
import matplotlib.pyplot as plt

import os
import json


# 读取json文件
# deberta reward model score
with open('./socre_deberta_0130_list.jsonl', "r", encoding="utf-8") as file:
    score_file = file.readlines()
    score_lines = [float(line.strip().strip("[").strip("]")) for line in score_file]
    
with open('./qwen_1p8_instag.jsonl', "r", encoding="utf-8") as file:
    tag_file_list = file.readlines()

with open("../hallucination_0201_full_40w.jsonl", 'r', encoding="utf-8") as file:
    hallu_score_lines = file.readlines()
    hallu_score_lines = [float(score) for score in hallu_score_lines]

with open("../socre_rerank_0201_list.jsonl", 'r', encoding="utf-8") as file:
    rank_score_lines = file.readlines()
    rank_score_lines = [float(score) for score in rank_score_lines]

with open("./socre_garbage_0205_list.jsonl", 'r', encoding="utf-8") as file:
    garbage_score_lines = file.readlines()
    garbage_score_lines = [json.loads(score) for score in garbage_score_lines]

# mixture/dj_mixture_challenge/deita_quality_40w/mixture_raw_reweight_task_type.json
with open("../deita_quality_40w/mixture_raw_reweight_task_type.json", "r",  encoding="utf-8") as file:
    deita_scores = json.load(file)
    
# /mnt/workspace/mixture/dj_mixture_challenge/output/sft_data/mixture_raw_reweight.jsonl
file1_path = '../output/sft_data/mixture_raw_reweight_task_type.jsonl'
output_path = '../output/sft_data/mixture_raw_reweight_task_type_add_new_tag_score.jsonl'

assert len(score_lines) == len(tag_file_list) == len(hallu_score_lines) == len(rank_score_lines)

import numpy as np
np.random.seed(42)
print_cnt = 200

score_cnt = 0
with open(file1_path, 'r') as file1 , open(output_path, 'w', encoding='utf-8') as outfile:
    # 使用zip_longest从两个文件中交替读取行，直到两个文件都读完
    # 如果其中一个文件比另一个长，它将继续读取，另一个文件的位置用None填充
    lines_ = file1.readlines()
    print(len(lines_))
    
    for idx, line in enumerate(lines_):
        cur_data = json.loads(line)
        
        
        cur_data["instag"] = tag_file_list[idx]
        score = score_lines[idx]
        cur_data["reward"] = score
        cur_data["deita_quality"] = deita_scores[idx]
        
        cur_data["hallucination"] = hallu_score_lines[idx]
        
        cur_data["rerank"] = rank_score_lines[idx]
        
        # if garbage_score_lines[idx][1] > 0.9:
        #     print(cur_data)
            
        
        outfile.write(json.dumps(cur_data, ensure_ascii=False) + '\n')
print(score_cnt)






import json
import pandas as pd
import matplotlib.pyplot as plt

import os
import json


# 读取json文件
# deberta reward model score
with open('./socre_deberta_0130_list.jsonl', "r", encoding="utf-8") as file:
    score_file = file.readlines()
    score_lines = [float(line.strip().strip("[").strip("]")) for line in score_file]
    
with open('./qwen_1p8_instag.jsonl', "r", encoding="utf-8") as file:
    tag_file_list = file.readlines()

with open("../hallucination_0201_full_40w.jsonl", 'r', encoding="utf-8") as file:
    hallu_score_lines = file.readlines()
    hallu_score_lines = [float(score) for score in hallu_score_lines]

with open("../socre_rerank_0201_list.jsonl", 'r', encoding="utf-8") as file:
    rank_score_lines = file.readlines()
    rank_score_lines = [float(score) for score in rank_score_lines]

with open("./socre_garbage_0205_list.jsonl", 'r', encoding="utf-8") as file:
    garbage_score_lines = file.readlines()
    garbage_score_lines = [json.loads(score) for score in garbage_score_lines]

print(len(garbage_score_lines))
# mixture/dj_mixture_challenge/deita_quality_40w/mixture_raw_reweight_task_type.json
with open("../deita_quality_40w/mixture_raw_reweight_task_type.json", "r",  encoding="utf-8") as file:
    deita_scores = json.load(file)
    
# /mnt/workspace/mixture/dj_mixture_challenge/output/sft_data/mixture_raw_reweight.jsonl
file1_path = '../output/sft_data/mixture_raw_reweight_task_type.jsonl'
output_path = '../output/sft_data/mixture_raw_reweight_task_type_add_new_tag_score.jsonl'

assert len(score_lines) == len(tag_file_list) == len(hallu_score_lines) == len(rank_score_lines)

import numpy as np
np.random.seed(42)
print_cnt = 100

score_cnt = 0
end_sign_cnt = 0
ghost_cnt=0

outfile_tmp = open("tmp_output.jsonl", 'w', encoding='utf-8')
    
generate_nonsense_count = 0
with open(file1_path, 'r') as file1:
    # 使用zip_longest从两个文件中交替读取行，直到两个文件都读完
    # 如果其中一个文件比另一个长，它将继续读取，另一个文件的位置用None填充
    lines_ = file1.readlines()
    print(len(lines_))
    
    for idx, line in enumerate(lines_):
        cur_data = json.loads(line)
        
        
        cur_data["instag"] = tag_file_list[idx]
        score = score_lines[idx]
        cur_data["reward"] = score
        cur_data["deita_quality"] = deita_scores[idx]
        
        cur_data["hallucination"] = hallu_score_lines[idx]
        
        cur_data["rerank"] = rank_score_lines[idx]
        
        if "</s> " in  cur_data["text"]:
            end_sign_cnt += 1
        if "ghost" in cur_data["text"]:
            # if print_cnt:
            #     print_cnt -= 1
            #     print(cur_data)
            ghost_cnt += 1
            
        if garbage_score_lines[idx][1] > 0.9:
            if print_cnt:
                print_cnt -= 1
                print(cur_data)
            
        # if "descriptive writing" in cur_data["instag"] or \
        #     "scene description" in cur_data["instag"] or \
        #     "set description" in cur_data["instag"] or \
        #     "character development" in cur_data["instag"] or \
        #     "plot development" in cur_data["instag"]:
        # if "imagery" in cur_data["instag"] or \
        #     "imagination" in cur_data["instag"] or \
        #     "fantasy" in cur_data["instag"]:
        # fiction write 
        # storytelling
        # ["编故事", "写作风格", "角色扮演", "解释", "故事续写"
        
        # role-playing
        # if topic_prob[0] > 0.40
        # if any(item in cur_data["topic_label"][:1] for item in ["编故事", "角色扮演", "故事续写"]):

            generate_nonsense_count += 1
            outfile_tmp.write(json.dumps(cur_data, ensure_ascii=False) + '\n')
print("end_sign_cnt",end_sign_cnt)
print("ghost_cnt",ghost_cnt)
print("generate_nonsense_count",generate_nonsense_count)





# 故事续写', '编故事

