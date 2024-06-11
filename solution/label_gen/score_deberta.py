import datasets
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
# from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


import os
import json
# import evaluate
# from evaluate import logging
from tqdm import tqdm

class Perplexity():
    def _compute(
        self, question_list, answer_list, model_id, batch_size: int = 16, add_start_token: bool = True, device=None, max_length=None, output_file_name=None
    ):

        if device is not None:
            assert device in ["gpu", "cpu", "cuda"], "device should be either gpu or cpu."
            if device == "gpu":
                device = "cuda"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
        model, tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name), AutoTokenizer.from_pretrained(reward_name)

        model = model.to(device)
        model.eval()
        
        ppls = []
        loss_list = []
        loss_fct = CrossEntropyLoss(reduction="none")
        with torch.no_grad(), open(os.path.join("", f"{output_file_name}"), "a") as w_file:
            for start_index in tqdm(range(0, len(question_list), batch_size)):
                end_index = min(start_index + batch_size, len(question_list))
                
                text_batch = question_list[start_index:end_index]
                answer_batch = answer_list[start_index:end_index]
                
                inputs = tokenizer(text_batch, answer_batch, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
                score = model(**inputs).logits.cpu().detach()


                ppls += score.tolist()
                
                for ppl_score in score.tolist():
                    w_file.write(str(ppl_score) + "\n")

        return {"deberta_score": ppls} # "mean_perplexity": loss_list}

file1_path = "../../output/sft_data/mixture_raw_reweight.jsonl"

output_file_name = "socre_deberta_0130_list.jsonl"

already_num = 0
try:
    with open(os.path.join("", f"{output_file_name}"), "r") as w_file:
        score_list = w_file.readlines()
        score_list = [s for s in score_list if len(str(s)) > 0]
        already_num = len(score_list)
except:
    pass

print("already_num", already_num)

with open(file1_path, 'r') as file1:
    lines_ = file1.readlines()[already_num:]
    lines_ = [json.loads(line) for line in lines_]
    question_list = [line["instruction"] + "\n" + line["input"] for line in lines_]
    answer_list = [line["output"] for line in lines_]

# question, answer = "Explain nuclear fusion like I am five", "Nuclear fusion is the process by which two or more protons and neutrons combine to form a single nucleus. It is a very important process in the universe, as it is the source of energy for stars and galaxies. Nuclear fusion is also a key process in the production of energy for nuclear power plants."

# question_list = [question]
# answer_list = [answer]
# print(lines_[:100])
# print(len(lines_))

# lines_ # lines_[:100]
# input_texts = lines_# ["lorem ipsum", "Happy Birthday!", "Bienvenue", "花开堪折直须折", "花开堪折直须折qweqweqw"]

model_id="./Qwen-1_8B" # ./Baichuan2-7B-Base
# ./Baichuan2-7B-Base
batch_size=8
add_start_token=True
max_length=512
device = "gpu"

per = Perplexity()
output_list = per._compute(question_list, answer_list, model_id, batch_size, add_start_token, None, max_length, output_file_name)

# output_file_name = "socre_deberta_0130.json"
# with open(os.path.join("", f"{output_file_name}"), "w") as w_file:
#     json.dump(output_list, w_file, ensure_ascii=False)