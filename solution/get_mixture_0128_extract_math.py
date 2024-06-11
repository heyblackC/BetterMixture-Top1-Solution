# =============================================================================
# 以下是一个采用随机采样的示例，参赛者可自由编写文件内容。
# =============================================================================

import json
import random
from pathlib import Path
import numpy as np

# from tools.tool_inputcheck import input_check

# 设置随机种子，保证结果可复现
seed = 42
random.seed(seed)

# 设置随机种子，以确保随机操作的可重复性
np.random.seed(42)
import math
from tqdm import tqdm
import timeout_decorator

# 如果使用 NumPy
# import numpy as np
# np.random.seed(seed)

# 如果使用 PyTorch
# import torch
# torch.manual_seed(seed)

# 开发套件根目录
base_dir = Path(__file__).resolve().parent.parent

import re
# regex_pattern = r"([-+]?[0-9]*\.?[0-9]+) *([-+*/]) *([-+]?[0-9]*\.?[0-9]+) *= *([-+]?[0-9]*\.?[0-9]+)"
# regex_pattern = r"([-+]?[\d\(\)]+[\.\d]* *[*/+\-\^] *[\.\,\d\ */+\-\^\(\)]+ *= *[-+]?[\d\w\(\)]+[\.\da-z]*(?: *[*/+\-\^]\s*[\d\w\(\)\.]+ *)*)"
# 匹配纯数字

'''
import re

# 定义正则表达式
regex = r"([-+]?[0-9]*\.?[0-9]+) *([-+*/]) *([-+]?[0-9]*\.?[0-9]+) *= *([-+]?[0-9]*\.?[0-9]+)"

# 需要匹配的字符串
text = "30 - 20 = 10"

# 进行匹配
match = re.match(regex, text)

if match:
    print("表达式匹配成功:", match.group(0))
    print("操作数1:", match.group(1))
    print("运算符:", match.group(2))
    print("操作数2:", match.group(3))
    print("结果:", match.group(4))
else:
    print("表达式不匹配")

Adding $2x$ to both sides gives $-7 = 9x + 2$. Subtracting 2 from both sides gives $-9 = 9x$. Dividing both sides by 9 gives $x = \boxed{-1}$. The answer is: -1	

'''

# 输入输出路径
input_dir = base_dir / "input"
ratio_path = base_dir / "output" / "sft_data" / "ratio.json"
mixture_path = base_dir / "output" / "sft_data" / "mixture_raw_math_only_regex.jsonl"

# 函数仅为示意，可自由改写
def generate_mixture(input_dir, ratio_path, mixture_path):
    # 假设采样 60k 条样本
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
    @timeout_decorator.timeout(5)
    def simple_search(regex_pattern, text):
        return re.search(regex_pattern, text)
    
    with open(mixture_path, "w", encoding="utf-8") as mixture_file:
        for file_path in file_list:
            file_name = file_path.split(r"/")[-1].strip().replace(".jsonl","")
            with open(file_path, "r") as f:
                samples = f.readlines()
                np.random.shuffle(samples)
                samples = [json.loads(line) for line in samples]
                print("正则清理数据前，条数：", len(samples))
                # samples = input_check(samples)
                # samples = [item for item in samples if (not item["html_block"]) and (not item["code_detect"]) and re.search(regex_pattern, item["text"])]
                samples_new = []
                regex_pattern = r"([-+]?[\d\(\)]+[\.\d]* *[*/+\-\^] *[\.\,\d\ */+\-\^\(\)]+ *= *[-+]?[\d\w\(\)]+[\.\da-z]*(?: *[*/+\-\^]\s*[\d\w\(\)\.]+ *)*)"
                for item in tqdm(samples):
                    try:
                        if simple_search(regex_pattern, item["output"]):
                            samples_new.append(item)
                    except:
                        pass
                        print("超时停止！")
                samples = samples_new
                
                # samples = [item for item in samples if re.search(regex_pattern, item["text"])]
                # prob = data_ratio_dict[file_name]
                # num_samples = int(prob * total_count)
                print(file_name)
                print(len(samples))
                selected_samples = samples
                for line in selected_samples:
                    mixture_file.write(json.dumps(line, ensure_ascii=False) + '\n')

def list_in_str(target_list, text):
    for target in target_list:
        if target in text:
            return True
    return False

# 执行函数

def validate_math_expression():
    print_cnt = 100
    file_path = "../output/sft_data/mixture_raw_math_only_regex.jsonl"
    output_path = "../output/sft_data/mixture_raw_math_only_regex_filtered.jsonl"
    
    with open(file_path, 'r', encoding="utf-8") as file, open(output_path, 'w', encoding="utf-8") as out_file:
        lines = file.readlines()
        lines = [json.loads(line) for line in lines]
        new_lines = math_filter_tool(lines, filter_higher_math=True, filter_long_number_data=True)
        
        for line in new_lines:
            out_file.write(json.dumps(line, ensure_ascii=False) + '\n')

def math_filter_tool(lines, filter_higher_math=True, filter_long_number_data=False):
    output_list = []
    def clean_expression(s):
        # 匹配数字、加减乘除运算符以及空格之外的任意字符
        pattern = r'^[^\d+\-*/ \.\%=\(\)a-zA-Z]*|[^\d+\-*/ \.\%=\(\)a-zA-Z]*$'
        # 删除不匹配的字符
        clean_s = re.sub(pattern, '', s)
        return clean_s
    from sympy import sympify, simplify
    # expr = sympify(str_expr)
    # expr
    regex_pattern = r"[-+]?\d[\.\,\d\w\ */+\-\^]+ *[*/+\-\^] *[\.\,\d\w\ */+\-\^]+ *= *[-+]?\d[\.\,\d\w\ */+\-\^]*"
    
    def check_equal(a_list, right_res):
        for left_res in a_list:
            if math.isclose(left_res, right_res, rel_tol=1e-2):
            # if abs(left_res - right_res) <= 1e-5:
                return True
        return False
    
    def detect_long_number(text, integer_bound=8, decimal_bound=7):
        long_float_number_pattern = r"[\d]*\.?[\d]+"
        results = re.findall(long_float_number_pattern, text)
        if results:
            max_integer_len = 0
            max_decimal_len = 0
            for res in results:
                splited_num = res.split(r".")
                if len(splited_num) == 2:
                    max_integer_len = max(len(splited_num[0]), max_integer_len)
                    max_decimal_len = max(len(splited_num[1]), max_decimal_len)
                elif len(splited_num) == 1:
                    max_integer_len = max(max_integer_len, len(splited_num[0]))
                else:
                    pass
            if max_integer_len >= integer_bound or max_decimal_len >= decimal_bound:
                return True
            else:
                return False
        return False
    
    def detect_comma_long_number(text,bound=8):
        long_number_pattern = r"[\d]+(?:\,[\d]+)+"
        results = re.findall(long_number_pattern, text)
        if results:
            max_len = 0
            for res in results:
                # print(res)
                max_len = max(len(res.replace(",","")),max_len)
            if max_len >= bound:
                return True
        return False
    
    def detect_sqrt_and_square(text):
        long_number_pattern = re.compile(r"[\d\w]+[²³⁴]|sqrt|√|[\d\w]+\^[-+]?[\d\w]+|log\([\d\w]*\)|log[\d]+|exp\([\d\w]*\)|cos\([\d\w]*\)|sin\([\d\w]*\)|Σ|frac\{.*\}|\d+i|\d+e[-+]?\d+|variance\(|[a-z]+\(.*\)", re.IGNORECASE)
        if re.search(long_number_pattern,text):
            return True
        if "quadratic equation" in text or "二次方程" in text:
            return True
        return False
    def check_bad_math_equation(text):
        equation_pattern = r"[-+]?[a-z\d\(\)]+[\.\da-z]* *[*/+\-] *[\.\,\d\w\ */+\-\^\(\)]+ *= *[-+]?[\d\w\(\)]+[\.\da-z]*(?: *[*/+\-] *[\d\w\(\)\.]+\s*)*"
        results = re.findall(equation_pattern, text)
        if results:
            for math_exp in results:
                # if not re.search(r"[a-zA-Z]", math_exp):
                #     continue
                left_equation, right_equation = math_exp.split("=")
                if re.search(r"[a-zA-Z]", left_equation) and re.search(r"[a-zA-Z]", right_equation):
                    try:
                        letters1 = set(re.findall(r'[a-zA-Z]', left_equation))
                        letters2 = set(re.findall(r'[a-zA-Z]', right_equation))
                        if len(letters1 - letters2) == 0:
                            expr1 = sympify(left_equation)
                            expr2 = sympify(right_equation)
                            # print(math_exp)
                            # print(expr1)
                            # print(expr2)
                            # simplify(left_equation + "-" + "(" + right_equation + ")")
                            if simplify(expr1 - expr2) != 0:
                                print("==========")
                                print(text)
                                print("出错了！！！！！")
                                print("expr1",expr1)
                                print("expr2",expr2)
                                return True
                    except:
                        pass
        return False
    def check_operation_result(regex_pattern, line, if_x=True):
        tmp_input_str = line["text"].replace("\\\\", "\\").replace(",","")
        
        def convert_percentage_to_decimal(text):
            pattern = re.compile(r'([-+]?\d*\.?\d+)%')
            def to_decimal(match):
                return str(float(match.group(1).replace("%","")) / 100)
            return pattern.sub(to_decimal, text)
        
        def remove_backslashes(text):
            # 匹配反斜杠后紧跟一个任意字符
            pattern = re.compile(r'\\([^\w])')
            # 用该字符替换匹配到的整个模式（即去掉反斜杠）
            return pattern.sub(r'\1', text)

        tmp_input_str = tmp_input_str.replace("$", " ")
        tmp_input_str = convert_percentage_to_decimal(tmp_input_str)
        tmp_input_str = remove_backslashes(tmp_input_str)
        tmp_input_str = re.sub(r'(\d+) *[x×] *(\d+)', r'\1 * \2', tmp_input_str)
        
        results = re.findall(regex_pattern, tmp_input_str)
        # print(results)
        if results:
            for math_exp in results:
                # print(math_exp)
                if "(" == math_exp[0] and ")" == math_exp[-1]:
                    math_exp = math_exp[1:-1]
                
                math_exp = math_exp.rstrip(r"\.").rstrip(r"\。")
                
                if not re.search(r"[a-zA-Z]", math_exp) and "=" in math_exp:
                    try:
                        left_half, right_half = math_exp.split("=")
                        _ = abs(eval(left_half) - eval(right_half))
                    except:
                        math_exp = clean_expression(math_exp)
                    left_half, right_half = math_exp.split("=")
                    left_half = left_half.replace(r'^', '**')
                    right_half = right_half.replace(r'^', '**')
                    try:
                        # print(left_half)
                        # print(right_half)
                        left_res = eval(left_half)
                        right_res = eval(right_half)

                        min_round = min(len(str(left_res).split(r".")[-1]), len(str(right_res).split(r".")[-1]))
                        # print(min_round)
                        
                        # if isinstance(left_res, float) and isinstance(right_res, float):
                        #     left_res = round(left_res, min_round)
                        #     right_res = round(right_res, min_round)
                        
                        # print(left_res)
                        # print(right_res)
                        gt_result = [left_res / 100, left_res, left_res * 100]
                        # print(gt_result)
                        if not (math.isclose(left_res, right_res, rel_tol=1e-2)) and not check_equal(gt_result, right_res):
                            if if_x and line["text"].count("x ") < 2: # and not re.search(r"[a-zA-Z][a-zA-Z]+ [*/+\-\x\×] \d+", line["text"]):
                                # print("=========================")
                                # print("原文：", line["text"])
                                # print("错误如下：", math_exp)
                                # print("检测到数学计算错误！！！！")
                                # print("=========================")
                                return True
                            else:
                                # print("=========================")
                                # print("原文：", line["text"])
                                # print("错误如下：", math_exp)
                                # print("检测到数学计算错误！！！！")
                                # print("=========================")
                                return True
                            # break
                    except:
                        pass
        return False
    
    # \b[-+]?[\.\,\d\w\ */+\-\^]+ *[*/+\-\^] *[\.\,\d\w\ */+\-\^]+ *= *[-+]?[\.\,\d\w\ */+\-\^]*\b
    wrong_cnt = 0
    long_number_cnt = 0
    comma_long_cnt = 0
    detect_sqrt_and_square_cnt = 0
    gsm8k_like_math_cnt = 0
    latext_math_cnt = 0
    sorry_cnt = 0 
    # with open(file_path, 'r', encoding="utf-8") as file, open(output_path, 'w', encoding="utf-8") as out_file:
    #     lines = file.readlines()
    
    for line in lines:
        # line = json.loads(line)
        # if line["html_block"]:
        #     continue
        
        if filter_long_number_data:
            if detect_long_number(line["text"]):
                long_number_cnt += 1
                # print(line["text"])
                continue

            if detect_comma_long_number(line["text"]):
                comma_long_cnt += 1
                # print(line["text"])
                continue
        
        # 以下是高等数学检测函数，不要保留高等数学计算，能提升效果
        if filter_higher_math and detect_sqrt_and_square(line["text"]):
            detect_sqrt_and_square_cnt += 1
            # print(line["text"])
            # print("===================")
            continue
            
        # if check_bad_math_equation(line["text"]):
        #     # print(line["text"])
        #     continue
        regex_pattern_2 = re.compile(r'[-+]?[a-z\d\(\)]+[\.\da-z]* *[*/+\-\^] *[\.\,\d\w\ */+\-\^\(\)]+ *= *[-+]?[\d\w\(\)]+[\.\da-z]*(?: *[*/+\-\^]\s*[\d\w\(\)\.]+ *)*', re.I)
        # if check_operation_result(regex_pattern,line):
        #     wrong_cnt += 1
        #     continue
        if check_operation_result(regex_pattern_2,line,if_x=False):
            wrong_cnt += 1
            continue

        regex_pattern_pure_number = re.compile(r'[\=\:] *([-+]?[\d\(\)]+[\.\d]* *[*/+\-\^] *[\.\,\d\ */+\-\^\(\)]+ *= *[-+]?[\d\w\(\)]+[\.\da-z]*(?: *[*/+\-\^]\s*[\d\w\(\)\.]+ *)*)', re.I)
        if check_operation_result(regex_pattern_pure_number,line,if_x=False):
            wrong_cnt += 1
            continue

        regex_pattern_last_result = re.compile(r'[\=\:] *([-+]?[\d\(\)]+[\.\d]* *[*/+\-\^] *[\.\,\d\ */+\-\^\(\)]+ *= *[-+]?\d*\.\d+\b)(?!\s*[+\-*/\^])')
        if check_operation_result(regex_pattern_last_result, line, if_x=False):
            wrong_cnt += 1
            # print(line["text"])
            continue

        regex_pattern_last_result_integer = r'[\=\:] *([-+]?[\d\(\)]+[\.\d]* *[*/+\-\^] *[\.\,\d\ */+\-\^\(\)]+ *= *[-+]?\d+\b)\.?[^\d\.+\-*/](?!\s*[+\-*/\^])'
        if check_operation_result(regex_pattern_last_result_integer, line, if_x=False):
            wrong_cnt += 1
            # print(line["text"])
            continue

        gsm8k_specific_detect_pattern = re.compile(r'<<[-+]?[a-z\d\(\)]+[\.\da-z]* *[*/+\-] *[\.\,\d\w\ */+\-\^\(\)]+ *= *[-+]?[\d\w\(\)]+[\.\da-z]*(?: *[*/+\-]\s*[\d\w\(\)\.]+ *)*>>', re.I)
        if re.search(gsm8k_specific_detect_pattern, line["text"]):
            gsm8k_like_math_cnt += 1
            # output_list.append(line)
        
        # 以下是latex过滤函数
        # latex_pattern = re.compile(
        #     r'\\[a-zA-Z]+{.*?}',re.IGNORECASE
        # )
        # if re.search(latex_pattern, line["text"]):
        #     latext_math_cnt += 1
        #     continue

        # delete_keywords = ["语言模型", "抱歉", "我无法", "Sorry", "sorry", "apologize", "我道歉", "我错误地",]
        # # "binary", "octonary", "hexadecimal", "coordinate", "坐标"
        # if any(keyword in line["output"][:30].lower() for keyword in delete_keywords):
        #     sorry_cnt += 1
        #     continue
        
        output_list.append(line)
        # out_file.write(json.dumps(line, ensure_ascii=False) + '\n')
        # if check_bad_math_equation(line["text"]):
        #     # print(line["text"])
        #     continue

    
    print("wrong_cnt",wrong_cnt)
    print("long_number_cnt",long_number_cnt)
    print("comma_long_cnt", comma_long_cnt)
    print("detect_sqrt_and_square_cnt",detect_sqrt_and_square_cnt)
    print("gsm8k_like_math_cnt",gsm8k_like_math_cnt)
    print("latext_math_cnt",latext_math_cnt)
    print("sorry_cnt",sorry_cnt)
    return output_list

# def generate_mixture_refine_math():
#     print_cnt = 100
#     file_path = "/mnt/workspace/mixture/dj_mixture_challenge/output/sft_data/mixture_raw_math_only.jsonl"
#     output_path = "/mnt/workspace/mixture/dj_mixture_challenge/output/sft_data/mixture_raw_math_only_refine.jsonl"
    
#     with open(file_path, 'r', encoding="utf-8") as file, open(output_path, 'w', encoding="utf-8") as out_file:
#         lines = file.readlines()
#         for line in lines:
#             line = json.loads(line)
#             task_type_list = line["topic_label"][:3]
#             if any(item in task_type_list for item in ["数学计算", "算法", "数组操作", "对比分析", "数据操作", "数据处理"]) or line["math_score"] >= 0.85:
#                 out_file.write(json.dumps(line, ensure_ascii=False) + '\n')
#             else:
#                 if print_cnt > 0:
#                     print_cnt -= 1
#                     print(line)

# generate_mixture(input_dir, ratio_path, mixture_path)
# generate_mixture_refine_math()

# generate_mixture(input_dir, ratio_path, mixture_path)
# validate_math_expression()