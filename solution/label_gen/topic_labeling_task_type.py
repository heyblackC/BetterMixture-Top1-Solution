import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm, trange

from torch.utils.data import Dataset
from tqdm import tqdm
import argparse
import re

import json
import transformers
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertModel,
    BertTokenizer,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
)

# from transformers import AutoModelForSequenceClassification, AutoTokenizer



import numpy as np
from sentence_transformers import SentenceTransformer


def topic_labels_closed():
    # file_path = "./subject_category.json"
    file_path = "./task_type.json"
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        label_dict = json.load(f)
        labels = label_dict["positive"] + label_dict["negtive"]
        print(len(labels))
        return labels
    return labels

class EvalDataset(Dataset):
    def __init__(self, tokenizer, data, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        # if self.data[index].get("instruction") == None:
        #     self.data[index]['instruction'] = "" + self.data[index].get("instruction", "") + \
        #                                    " "+ self.data[index].get("input", "") + \
        #                                    "\n" + self.data[index].get("output", "")
        query = self.data[index]['instruction']
        query = str(query)

        return query

def evaluate(phrase_list, batch_size):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    closed_labels = topic_labels_closed()
    closed_labels = [label for label in closed_labels] # "为这个句子生成表示以用于检索相关文章：" +
    closed_labels = np.array(closed_labels)

    model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
    
#     tokenizer = AutoTokenizer.from_pretrained('./bge-reranker-base')
#     model = AutoModelForSequenceClassification.from_pretrained('./bge-reranker-base')
    
    # use gpu
    # model.half()
    print(device)
    model.to(device)
    model.eval()

    eval_dataset = EvalDataset(None, phrase_list, 512)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=batch_size
    )
    with torch.no_grad():
        embeddings_2 = model.encode(closed_labels, normalize_embeddings=True)
    
    pred_labels=[]
    pred_values=[]
    
    preds = None
    for source in tqdm(eval_dataloader):
        source_new = []
        for line_s in source:
            tokens = tokenizer.encode(line_s)
            if len(tokens) > 512:
                segment = tokens[:256] + tokens[-256:]
                segment_text = tokenizer.decode(segment)
                source_new.append(segment_text)
                print("trigger!!")
            else:
                source_new.append(line_s)
        
        model.eval()
        with torch.no_grad():
            
            embeddings_1 = model.encode(source, normalize_embeddings=True)
            
            similarity = embeddings_1 @ embeddings_2.T

            if preds is None:
                preds = similarity
            else:
                preds = np.append(preds, similarity, axis=0)

            # pred_values.extend(max_values)
            # pred_labels.extend(predict_lables)
    
    return preds


def read_sft_datas(cleaned_input_dir, bert_classification_high_dir, is_wudao=True):

    
    file_name_list = os.listdir(cleaned_input_dir)
    file_name_list = [file_name for file_name in file_name_list if file_name[0] != '.' and ".jsonl" in file_name]
    file_name_list = list(sorted(file_name_list))

    os.makedirs(bert_classification_high_dir, exist_ok=True)

    high_output_dir = os.path.join(bert_classification_high_dir, "label")
    low_output_dir = os.path.join(bert_classification_high_dir, "low")
    
    os.makedirs(high_output_dir, exist_ok=True)
    os.makedirs(low_output_dir, exist_ok=True)

    filtered_name_list = os.listdir(low_output_dir) + os.listdir(high_output_dir)
    
    check_file = os.path.join(bert_classification_high_dir, "check_file.txt")
    
    if not os.path.exists(check_file):
        with open(check_file, 'w', encoding='utf-8') as check:
            pass

    for wudao_file in file_name_list:
        if wudao_file in filtered_name_list:
            continue

        with open(os.path.join(cleaned_input_dir, wudao_file), 'r', encoding='utf-8') as f:
            
            thresh = 0.50
            
            with open(check_file, 'r', encoding='utf-8') as check:
                check_lines = check.readlines()
                check_lines = " ".join(check_lines)
                if check_lines == None:
                    check_lines = ""
                if str(wudao_file) in str(check_lines):
                    continue
            
            with open(check_file, 'a', encoding='utf-8') as check:
                check.write(wudao_file + '\n')
            
            # data = json.load(f)
            lines = f.readlines()
            data = [json.loads(line) for line in lines]
            print(len(data))
            pred_values = evaluate(data, 1024)
            print(len(pred_values))
            # bsz, label_size
            
            high_list = []
            low_list = []
            
            closed_labels = topic_labels_closed()
            
            print("开始打标签---")
            for idx in tqdm(range(len(data))):
                
                # if data[idx]["dataType"] == "百科":
                #     thresh = 0.6
                # else:
                
                eval_score = pred_values[idx]
                top3_indices = np.argsort(-eval_score)
                # top3_indices = [idx for idx in top3_indices if eval_score[idx]>thresh]
                top3_indices = top3_indices[:10]
                
                entity = data[idx]
                
                if len(top3_indices) > 0:
                    pre_label_list = list(np.array(closed_labels)[top3_indices])
                    entity["topic_label"] = pre_label_list
                    entity["topic_prob"] = [float(eval_score[position_idx]) for position_idx in top3_indices]
                    high_list.append(entity)
                # else:
                #     entity["topic_label"] = [str("其他")]
                #     entity["prob"] = []
                #     low_list.append(entity)

            with open(os.path.join(high_output_dir, wudao_file), "w", encoding='utf-8') as w_file:
                for line in high_list:
                    w_file.write(json.dumps(line, ensure_ascii=False) + '\n')

                # json.dump(high_list, w_file, ensure_ascii=False)
            # with open(os.path.join(low_output_dir, wudao_file), "w") as w_file:
            #     json.dump(low_list, w_file, ensure_ascii=False)

            print(str(wudao_file))
            print("标签数量：" + str(len(high_list)))
            print("去掉数据数量：" + str(len(low_list)))

# read_sft_datas(is_wudao=True)

def main():
    
    parser = argparse.ArgumentParser()

    # 添加选项和参数
    parser.add_argument('--cleaned_input_dir', type=str, default='/mnt/workspace/peipao/datasets/pretrain/WuDao200G_cleaned', help='input dir')
    parser.add_argument('--output_dir', type=str, default="/mnt/workspace/peipao/datasets/pretrain/quality_filter_bert_0918", help='output dir')


    args = parser.parse_args()
    
    print(args)
    
    
    read_sft_datas(args.cleaned_input_dir, args.output_dir)

if __name__ == '__main__':
    main()
