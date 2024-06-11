import json
import re
import os
# import evaluate
# from lexicalrichness import LexicalRichness
from tqdm import tqdm

# toxicity_tool = evaluate.load("toxicity", module_type="measurement")

# dataset = "alpaca_data.json"
# # Load JSON data
# with open(dataset, "r") as f:
#     json_data = json.load(f)

# regex for "<noinput>", "No Input", "<No input>", "noinput", "<no input>"

'''
def add_space_between_emojies(text):
  # Ref: https://gist.github.com/Alex-Just/e86110836f3f93fe7932290526529cd1#gistcomment-3208085
  # Ref: https://en.wikipedia.org/wiki/Unicode_block
  EMOJI_PATTERN = re.compile(
    "(["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "])"
  )
  text = re.sub(EMOJI_PATTERN, r' \1 ', text)
  return text
https://gist.github.com/Alex-Just/e86110836f3f93fe7932290526529cd1
'''
noinput_pattern = re.compile(r"[\[\(\<]?no[ ]?input[\]\)\>\.]?", re.IGNORECASE)

# regex for "![any string](http" to detect if internet data is being passed into input
img_pattern = re.compile(r"!\[.*\]\(http|img[ ]?src=|image:")

http_link_pattern = re.compile(r"(http(s)?:\/\/)[\w.-]+(?:\.[\w\.-]+)+[\w\-\._~:/?#[\]@!\$&'\(\)\*\+,;=.]+")

# regex for locating 'merged' instructions. Seem to mostly be in the output
merged_pattern = re.compile(r"\d{1,2}\.\sInstruction:", re.IGNORECASE)

angle_brackets_pattern = re.compile(r"^<[^>]+>$", re.IGNORECASE)

nooutput_pattern = re.compile(r"[\[\(\<]?no[ ]?output[\]\)\>\.]?", re.IGNORECASE)

wrong_indices_pattern = re.compile("\n1\. [^2]*\n1\. ", re.DOTALL)


div_pattern = re.compile("<div.*?>")

span_pattern = re.compile("<span.*?>")

code_lang_pattern = re.compile(
    r"```\s*" + "(.*?)" + "(?:Copy code)+" + "(.+?)" + "\s*?```", re.DOTALL,
)

code_detect_pattern = re.compile(r"`{3}[^\n]+?\n(.*?[^`]+)`{3}", re.IGNORECASE)

code_detect_more_loose = re.compile(r'```(.*?)```', re.DOTALL)

# html_re_pattern = re.compile(r'<(\S?)[^>]>.?|<.*?\/>',re.DOTALL)
html_re_pattern = re.compile(r'<!--.*?-->|<([a-zA-Z][a-zA-Z0-9]*)\b[^>]*>([\s\S]*?)<\/\1>')

# java_pack_pattern = re.compile("([a-zA-Z_]\w*)+([.][a-zA-Z_]\w*)+")

punc_pattern = re.compile(r'[\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b\n]')

EMOJI_PATTERN = re.compile(
"(["
"\U0001F1E0-\U0001F1FF"  # flags (iOS)
"\U0001F300-\U0001F5FF"  # symbols & pictographs
"\U0001F600-\U0001F64F"  # emoticons
"\U0001F680-\U0001F6FF"  # transport & map symbols
"\U0001F700-\U0001F77F"  # alchemical symbols
"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
"\U0001FA00-\U0001FA6F"  # Chess Symbols
"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
"\U00002702-\U000027B0"  # Dingbats
"])"
)

question_begin_pattern = re.compile("(which|who|what|how|when|where|why) ", re.IGNORECASE)

options_pattern = re.compile("a\..*b\..*c\..*d\.", re.IGNORECASE)

qa_pair_pattern = re.compile("(question|q):\s*(.*?)\s*(answer|a):\s*", re.IGNORECASE) # 

def contains_unwanted_words(text):
    unwanted_words = [
        "prioritize human safety"
        "ethical principles"
        "harmful to human beings"
        "September 2021"
        "as a language model",
        "ethical guidelines",
        "as an AI language model",
        "my guidelines",
        "As an AI",
        "prioritize user safety",
        "adhere to ethical guidelines",
        "harmful consequences",
        "potentially harmful",
        "dangerous activities",
        "promote safety",
        "well-being of all users",
        "responsible information sharing",
        "jeopardize the safety",
        "illegal actions or intentions",
        "undermine the stability",
        "promote the well-being",
        "illegal activities or actions",
        "adherence to the law",
        "potentially be harmful",
        "illegal substances or activities",
        "committed to promoting",
        "safe information",
        "lawful information",
        "cannot provide guidance",
        "cannot provide information",
        "unable to offer assistance",
        "cannot engage in discussions",
        "programming prohibits",
        "follow ethical guidelines",
        "ensure the safety",
        "involves an illegal subject",
        "prioritize safety",
        "illegal subject",
        "prioritize user well-being",
        "cannot support or promote",
        "activities that could harm",
        "pose a risk to others",
        "against my programming",
        "activities that could undermine",
        "potentially dangerous",
        "not within the scope",
        "designed to prioritize safety",
        "not able to provide",
        "maintain user safety",
        "adhere to safety guidelines",
        "dangerous or harmful",
        "cannot provide any information",
        "focus on promoting safety"
    ]
    for word in unwanted_words:
        if word.lower() in text.lower():
            return True
    return False

# `{3}[^\n]+?\n(.*?[^`]+)`{3}
# ^<[^>]+>$
# ([\d\^\+\-\/yx=,\(\) ]|\\u00b\d)+


import fasttext
from .text_normalizer import normalize

def score_text(model, text):
    normalized_text = normalize(text).replace('\n', ' ')
    # Remove any [EQUATION] tokens
    normalized_text = normalized_text.replace('[EQUATION]', '')
    pred = model.predict(normalized_text, k=2)
    if pred[0][0] == '__label__positive':
        prob = pred[1][0]
    else:
        prob = pred[1][1]

    return prob
from pathlib import Path
base_dir = Path(__file__).resolve().parent

math_model = fasttext.load_model(os.path.join(base_dir, 'math_score.bin'))
print(os.path.join(base_dir, 'math_score.bin'))

def input_check(json_data):
    filtered_list = []
    issue_cnt_1 = 0
    issue_cnt_2 = 0
    issue_cnt_3 = 0
    issue_cnt_4 = 0
    issue_cnt_5 = 0
    issue_cnt_6 = 0
    issue_cnt_7 = 0
    issue_cnt_8 = 0
    issue_cnt_9 = 0
    issue_cnt_10 = 0
    issue_cnt_11 = 0
    html_block_cnt = 0
    
    # Loop through JSON data and output items that contain "input" elements matching the regex
    print("<noinput> problems:")
    print("![alt text] problems:")
    total_print = 1000


    for item in tqdm(json_data):
        if "input" in item and noinput_pattern.search(item["input"].lower()):
            # print(item)
            issue_cnt_1 += 1
            continue
        if "input" in item and img_pattern.search(item["instruction"] + "\n" +item["input"] + item["output"]):
            # print(item) 
            issue_cnt_2 += 1
            continue
        if "output" in item and merged_pattern.search(item["output"]):
            # print(item)
            issue_cnt_3 += 1
            continue
        
        if angle_brackets_pattern.search(item["output"]) or angle_brackets_pattern.search(item["input"]):
            issue_cnt_4 += 1
            # print(item)
            continue
        if nooutput_pattern.search(item["output"]):
            issue_cnt_5 += 1
            # print(item)
            continue
        if re.search(wrong_indices_pattern, item["output"]):
            # print(item)
            issue_cnt_6 += 1
            continue
        if re.search(div_pattern, item["output"]) or re.search(span_pattern, item["output"]) or re.search(code_lang_pattern, item["output"]) or re.search(code_detect_pattern, item["output"]):
            # if re.search(code_detect_pattern, item["output"]):
            #     if total_print > 0:
            #         total_print -= 1
            #         print(item)
            # print(item)
            issue_cnt_7 += 1
            continue
        if contains_unwanted_words(item["output"]):
            issue_cnt_8 += 1
            # continue
        if re.search(html_re_pattern, item["text"]):
            # if total_print > 0:
            #     total_print -= 1
            #     print(item)
            html_block_cnt += 1
            item["html_block"] = True
        else:
            item["html_block"] = False
            

        # filter_words = ["继续", "接着写", "接着说", "Continue", "continue"]        
        # if any(keyword in item["instruction"][:15] for keyword in filter_words):
        #     if not any(keyword in item["instruction"] for keyword in ('for', 'while')):
        #         issue_cnt_9 += 1
        #         if item["meta"]["Multi-round Dialog"] != "True" and len(item["input"]) <= 0:
        #             print(item)
        #         continue
        
        delete_keywords = ["语言模型", "抱歉", "我无法", "Sorry", "sorry", "apologize", "language model"]
        if any(keyword in item["output"][:20] for keyword in delete_keywords):
            issue_cnt_9 += 1
            # continue
        if re.search(EMOJI_PATTERN, item["instruction"] + "\n" +item["input"] + item["output"]):
            issue_cnt_10 +=1
            # print(item)
            continue
        if re.search(http_link_pattern, item["text"].replace("\\", "")):
            issue_cnt_11 += 1
            continue
        # if not re.match(punc_pattern, item["output"].strip()[-1:]):
        #     print(item)
        #     issue_cnt_11 += 1
        #     continue
        if re.search(code_detect_more_loose, item["instruction"] + "\n" + item["input"] + item["output"]):
            issue_cnt_7 += 1
            item["code_detect"] = True
        else:
            item["code_detect"] = False
        
        math_score_tmp = score_text(math_model, item["instruction"] + "\n" + item["input"] + item["output"])
        item["math_score"] = math_score_tmp # 使用阈值0.70

        
        # toxi_results = toxicity_tool.compute(predictions=[item["instruction"] + "\n" + item["input"], item["output"]], aggregation="maximum")
        # item["toxicity_score"] = round(toxi_results['max_toxicity'], 4)
        
        whole_text = item["instruction"] + "\n" + item["input"] + item["output"]
        # if item["meta"]["Lang"] == "EN" and len(whole_text) > 0:
        #     try:
        #         whole_richness = LexicalRichness(whole_text)
        #         item["mtld_score"] = whole_richness.mtld()
        #     except:
        #         item["mtld_score"] = 0
        # else:
        #     item["mtld_score"] = 0

        # whole_text = item["instruction"] + "\n" + item["input"] + item["output"]
        item["line_break_cnt"] = item["text"].count("\n") #暂不使用
        item["o_i_ratio"] = round(len(item["output"])/(len(item["instruction"]+item["input"])+0.001),3) # 暂不使用
        
        item["sql_detect_cnt"] = item["text"].count("SELECT") + item["text"].count("FROM") # >=2 来使用
        # item["code_def_return_cnt"] = item["text"].count("def ") + item["text"].count("return") + item["text"].count("print") # 直接删除，没用
        
        item["Article_contain"] = True if "Article:" in item["text"] else False # 直接删除
        
        item["abcd_options"] = True if re.search(options_pattern, item["instruction"] + item["input"]) else False
        
        item["seven_ask"] = True if re.search(question_begin_pattern, item["text"][:20].strip()) else False
        
        item["qa_pair"] = True if re.search(qa_pair_pattern, item["text"].strip()) else False
        

        filtered_list.append(item)

    print(f"Identified {issue_cnt_1} potential <noinput> issues.")

    # issue_cnt = 0
    # Loop through JSON data and output items that contain "input" elements matching the regex
    print(f"Identified {issue_cnt_2} potential ![alt text] issues.")
    
    print(f"不同对话融合在一起：{issue_cnt_3}")
    print(f"尖括号内容：{issue_cnt_4}")
    print(f"nooutput输出：{issue_cnt_5}")
    print(f"编号错误内容：{issue_cnt_6}")
    print(f"代码输出和html格式输出：{issue_cnt_7}")
    print(f"AI风格触发词，不过滤：{issue_cnt_8}")
    print(f"中英文抱歉数量，不过滤：{issue_cnt_9}")
    print(f"过滤emogji数据数量：{issue_cnt_10}")
    print(f"过滤的超链接输出输入数量：{issue_cnt_11}")
    print(f"html block检测：{html_block_cnt}")
    return filtered_list