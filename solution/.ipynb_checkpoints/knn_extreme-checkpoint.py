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

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__file__)

# mixture/dj_mixture_challenge/output/sft_data_0119backup

def run():
    print_cnt = 1000
    sent_model = SentenceTransformer("maidalun1020/bce-embedding-base_v1")
    print(sent_model)
    sent_model.to(torch.device("cuda:0"))
    print(sent_model.device)
    
    file1_path = "./task_type_mixback_new_keep_cn.jsonl"
    output_file = "./task_type_mixback_new_keep_cn_knn_filtered.jsonl"

    with open(file1_path, 'r') as file1:
        lines_ = file1.readlines()
        raw_data = [json.loads(line) for line in lines_]
        raw_data = sorted(raw_data, key=lambda x: x['deita_quality'], reverse=True)
        
        # lines_ = sorted(lines_, key=lambda x: x['math_score'], reverse=True)
        db_dataset = [line["text"] for line in raw_data]


    db_embeddings = sent_model.encode(db_dataset, show_progress_bar=True)
    print("start nndesent", len(db_embeddings))
    
    
#     # 使用t-SNE进行降维处理
#     tsne = TSNE(n_components=2, random_state=42)  # 降到二维空间
#     tsne_embeddings = tsne.fit_transform(db_embeddings)

#     # 将降维后的结果转换为DataFrame以便于绘图
#     df = pd.DataFrame(tsne_embeddings, columns=['x', 'y'])

#     # 可视化
#     sns.scatterplot(x="x", y="y", data=df)
#     plt.show()
    

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
run()