from vllm import LLM, SamplingParams
from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import load_model, get_conversation_template
import torch
import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer



os.environ['VLLM_USE_MODELSCOPE']='False'

model_id = "qwen-1p8b-tagger"

# 使用的是https://www.modelscope.cn/models/lukeminglkm/instagger_qwen1_8B/summary
# InsTag指令打标工具千问1.8B版本，自行下载
tokenizer = AutoTokenizer.from_pretrained('../../instagger_qwen1_8B',trust_remote_code=True)



# Load model with vLLM
model = LLM(model="../../instagger_qwen1_8B", trust_remote_code=True)

# Setting greedy decoding with temperature=0
sampling_params = SamplingParams(temperature=0, max_tokens=512)

torch.manual_seed(0)

file1_path = "../../output/sft_data/mixture_raw_reweight.jsonl"

with open(file1_path, 'r') as file1:
    lines_ = file1.readlines()
    new_line_list = []
    for line in tqdm(lines_):
        line = json.loads(line)
        ins_text = line["instruction"]
        tokens = tokenizer.encode(ins_text)
        if len(tokens) >= 2000:
            tokens = tokens[:2000]
        cutoff_text = tokenizer.decode(tokens)
        line["instruction"] = cutoff_text
        new_line_list.append(line)

lines_ = new_line_list
# print(lines_[:100])
print(len(lines_))
# lines_ = lines_[:1000]

prompts = []
for line in lines_:
    conv = get_conversation_template(model_id)
    instruction = line["instruction"] if len(line["instruction"]) > 1 else line["instruction"] + " " + line["input"]
    conv.append_message(conv.roles[0], instruction)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    prompts.append(prompt)

outputs = model.generate(prompts, sampling_params)

output_path = "./qwen_1p8_instag.jsonl"
with open(output_path, 'w', encoding='utf-8') as outfile:
    for output in outputs:
        output_ids = output.outputs[0].token_ids

        # be consistent with the template's stop_token_ids
        if conv.stop_token_ids:
            stop_token_ids_index = [
                i
                for i, id in enumerate(output_ids)
                if id in conv.stop_token_ids
            ]
            if len(stop_token_ids_index) > 0:
                output_ids = output_ids[: stop_token_ids_index[0]]

        output = model.get_tokenizer().decode(
            output_ids,
            spaces_between_special_tokens=False,
        )
        if conv.stop_str and output.find(conv.stop_str) > 0:
            output = output[: output.find(conv.stop_str)]

        for special_token in model.get_tokenizer().special_tokens_map.values():
            if isinstance(special_token, list):
                for special_tok in special_token:
                    output = output.replace(special_tok, "")
            else:
                output = output.replace(special_token, "")
        output = output.strip()
        outfile.write(output + '\n')