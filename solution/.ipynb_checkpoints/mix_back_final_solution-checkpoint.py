import json
import os
file1_path = '../output/sft_data/juicer_out/task_type_mixback_new_keep_cn_knn_filtered.jsonl'


output_path = '../output/sft_data/mixture.jsonl'
import numpy as np
np.random.seed(42)
from tqdm import tqdm
import re

regex_pattern = r"[-+]?[0-9]*\.?[0-9]+(?: *[-+*/] *[0-9]*\.?[0-9]+)* *= *[-+]?[0-9]*\.?[0-9]+(?: *[-+*/] *[0-9]*\.?[0-9]+)*"

# from tools.tool_inputcheck import contains_unwanted_words

from transformers import AutoTokenizer

model_path = "baichuan-inc/Baichuan2-7B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

total_tokens = 1.0005e7 * 0.75
final_total_tokens = 1.0e7
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
o_i_ratio_cnt = 0

append_files_path = "../output/sft_data/math_scorer_juicer/mixture_raw_math_only_juicer_task_type.jsonl"
append_file = open(append_files_path, 'r', encoding='utf-8')
append_file_list = append_file.readlines()
append_file_list = [json.loads(line) for line in append_file_list]
from get_mixture_0128_extract_math import math_filter_tool
append_file_list = math_filter_tool(append_file_list)

math_high_score_cnt = 0
cache_list = []
for line in append_file_list:
    # line = json.loads(line)
    if line["math_score"] < 0.80:
        continue
    if line["html_block"] or line["code_detect"]:
        continue
    if r"</s>" in line["text"]:
        continue
    task_type_list = line["topic_label"][:3]
    if any(item in task_type_list for item in ["数学计算", "算法", "数组操作", "数据操作", "数据处理"]) and not re.search(regex_pattern, line["text"]):
        tokens = tokenizer.encode(line["text"])
        if len(tokens) > 4080:
            continue
        total_tokens -= len(tokens)
        cache_list.append(line)
print("cache_list", len(cache_list))

append_files_path2 = "../output/sft_data/mixture_raw_math_only_regex_filtered.jsonl"
append_file2 = open(append_files_path2, 'r', encoding='utf-8')
append_file_list_2 = append_file2.readlines()
append_file_list_2 = [json.loads(line) for line in append_file_list_2]
for line in append_file_list_2:
    tokens = tokenizer.encode(line["text"])
    if len(tokens) > 4080:
        continue
    if r"</s>" in line["text"]:
        continue
    total_tokens -= len(tokens)
    cache_list.append(line)

print("cache_list", len(cache_list))

print("total_tokens", total_tokens)
# deberta_score_list_path = "./socre_deberta_0130_list.jsonl"
# score_file = open(deberta_score_list_path, 'r', encoding='utf-8')
# score_file = score_file.readlines()
# score_lines = [line.strip().strip("[").strip("]") for line in score_file]
# print("score_lines", len(score_lines))

AI_language_out_cnt = 0
sorry_cnt = 0
code_detect_cnt = 0
# 8000
bins_cnt = 6000//100
distribute_count_list = [8000]*bins_cnt
print(distribute_count_list)
output_distribute_count_list = [9000]*bins_cnt
print(output_distribute_count_list)
duplicate_line_cnt = 0
HC3_Chinese_Human_cnt = 0
instag_code_filter_cnt = 0
direct_add_cnt_sum = 0
too_short_filter_cnt = 0

output_final_list = []

with open(file1_path, 'r', encoding="utf-8") as file1, open(output_path, 'w', encoding='utf-8') as outfile:
    
    
    np.random.shuffle(cache_list)
    for line in cache_list:
        # if line["meta"]["original_path"] == "Alpaca-CoT/HC3/HC3_Chinese_Human.json":
        #     HC3_Chinese_Human_cnt += 1
        #     continue
        
        if line.get("__dj__stats__",None):
            del line["__dj__stats__"]
        if line.get("topic_label",None):
            del line["topic_label"]
            del line["topic_prob"]
        if line.get("knn_n", None):
            del line["knn_n"]
        line.pop('html_block', None)
        line.pop('code_detect', None)
        line.pop('code_detect', None)
        line.pop('math_score', None)
        line.pop('line_break_cnt', None)
        line.pop('o_i_ratio', None)
        line.pop('sql_detect_cnt', None)
        line.pop('Article_contain', None)
        line.pop('abcd_options', None)
        line.pop('seven_ask', None)
        line.pop('qa_pair', None)
        line.pop('instag', None)
        line.pop('reward', None)
        line.pop('rerank', None)
        line.pop('hallucination', None)
        line.pop('deita_quality', None)



        output_final_list.append(line)
        # outfile.write(json.dumps(line, ensure_ascii=False) + '\n')
        
    # 使用zip_longest从两个文件中交替读取行，直到两个文件都读完
    # 如果其中一个文件比另一个长，它将继续读取，另一个文件的位置用None填充
    lines_ = file1.readlines()
    lines_ = [json.loads(line) for line in lines_]
    
    lines_ = math_filter_tool(lines_, filter_higher_math=False)
    # filter_higher_math
    
    def contains_chinese(text):
        chinese_pattern = re.compile(r'[\u4e00-\u9fa5]')
        return bool(chinese_pattern.search(text))
    
#     tmp_line_list = []
#     for line in tqdm(lines_):
#         if not contains_chinese(line["text"]) and len(line["output"]) < 200:
#             continue
        
#         if contains_chinese(line["text"]) and len(line["output"]) < 50:
#             continue
        
#         if any(item in line["instag"] for item in ["math", "calculation", "arithmetic"]):
#             output_final_list.append(line)
#             direct_add_cnt_sum += 1
#             continue
#         tmp_line_list.append(line)
    
#     lines_ = tmp_line_list

    # np.random.shuffle(lines_)
    lines_ = sorted(lines_, key=lambda x: x['deita_quality'], reverse=True)

    # prompt_toxi_detect = []
    print(len(lines_))
    for line in tqdm(lines_):
        # line = json.loads(line)
        if r"</s>" in line["text"]:
            continue

        if not contains_chinese(line["text"]) and len(line["output"]) < 200:
            too_short_filter_cnt += 1
            continue
        if contains_chinese(line["text"]) and len(line["output"]) < 50:
            too_short_filter_cnt += 1
            continue

        # if line["meta"]["original_path"] == "Alpaca-CoT/GPTeacher/GPTeacher.json":
        #     HC3_Chinese_Human_cnt += 1
        #     continue

#         if line["meta"]["original_path"] == "Alpaca-CoT/HC3/HC3_Chinese_Human.json":
#             HC3_Chinese_Human_cnt += 1
#             # print(line)
#             continue
        
#         # any(item in topic_lables_need for item in ['SQL生成', '数据库操作', '代码修改', '代码生成',]):
#         if any(item in line["instag"] for item in ["java", "html", "c++", "python", "php", "sql", "web development", "error handling", "software development", "programming", "code execution", "debug", "data visualization"]):
#             instag_code_filter_cnt += 1
#             if print_cnt > 0:
#                 print_cnt -= 1
#                 print(line)
#             continue

        # instag_list = cur_data["instag"]
        # instag_list = [tag for tag in instag_list if len(tag) > 3]
        # all_instag_string = ",".join(instag_list)
            
        # if line["o_i_ratio"] >= 14:
        #     continue

        # if line["o_i_ratio"] > 9:
        #     o_i_ratio_cnt += 1
        #     continue
        
        # bin_index =  len(line["instruction"] + line["input"]) // 100
        
        # if distribute_count_list[bin_index] > 0:
        #     distribute_count_list[bin_index] -= 1
        # else:
        #     continue
        # output_bin_index = len(line["output"]) // 100
        
        # if output_distribute_count_list[output_bin_index] > 0:
        #     output_distribute_count_list[output_bin_index] -= 1
        # else:
        #     continue

        if line["seven_ask"]:
            seven_ask_cnt += 1
        tokens = tokenizer.encode(line["text"])
        total_tokens -= len(tokens)
        if total_tokens < 0:
            print("跳出!!")
            break
        if line.get("__dj__stats__",None):
            del line["__dj__stats__"]
        if line.get("topic_label",None):
            del line["topic_label"]
            del line["topic_prob"]
        if line.get("reward",None):
            del line["reward"]
        if line.get("knn_n", None):
            del line["knn_n"]
        line.pop('html_block', None)
        line.pop('code_detect', None)
        line.pop('code_detect', None)
        line.pop('math_score', None)
        line.pop('line_break_cnt', None)
        line.pop('o_i_ratio', None)
        line.pop('sql_detect_cnt', None)
        line.pop('Article_contain', None)
        line.pop('abcd_options', None)
        line.pop('seven_ask', None)
        line.pop('qa_pair', None)
        line.pop('instag', None)
        line.pop('reward', None)
        line.pop('rerank', None)
        line.pop('hallucination', None)
        line.pop('deita_quality', None)

        output_final_list.append(line)
        # outfile.write(json.dumps(line, ensure_ascii=False) + '\n')
    # np.random.shuffle(output_final_list)
    seen_prompts = set()
    seen_outputs = set()

    final_dedup_list = []
    for line in output_final_list:
        input_prompt = line["instruction"] + line["input"]
        if input_prompt not in seen_prompts and line["output"] not in seen_outputs:
            seen_prompts.add(input_prompt)
            seen_outputs.add(line["output"])
            final_total_tokens
            tokens = tokenizer.encode(line["text"])
            final_total_tokens -= len(tokens)
            if final_total_tokens < 0:
                print("跳出!!")
                break
            final_dedup_list.append(line)
            # outfile.write(json.dumps(line, ensure_ascii=False) + '\n')
        else:
            duplicate_line_cnt += 1
    
    np.random.shuffle(final_dedup_list)
    for line in final_dedup_list:
        outfile.write(json.dumps(line, ensure_ascii=False) + '\n')

print(distribute_count_list)
print(output_distribute_count_list)
print("o_i_ratio_cnt", o_i_ratio_cnt)
print("seven_ask_cnt",seven_ask_cnt)
print("duplicate_line_cnt",duplicate_line_cnt)
print("HC3_Chinese_Human_cnt", HC3_Chinese_Human_cnt)
print("instag_code_filter_cnt", instag_code_filter_cnt)
print("direct_add_cnt_sum", direct_add_cnt_sum)
print("too_short_filter_cnt", too_short_filter_cnt)
