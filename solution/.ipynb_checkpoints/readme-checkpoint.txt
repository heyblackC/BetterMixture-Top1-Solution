1.程序主文件和入口：
由于./tools/math_score.bin不存在，需要下载，请运行gen_data.sh!!!该文件会自动下载math_score.bin
请运行gen_data.sh!!!！！
get_mixture.py或者gen_data.sh

2.生成mixture.jsonl文件的流程如下：

# 第一步：混合原始数据
generate_mixture(input_dir, ratio_path, mixture_path)

# 第二步：输出混合ratiodict
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

3.前置依赖标签文件生成（可选，不是必跑项）
请注意，insert_new_param()和insert_new_param_for_math_score()依赖文件列表如下：
socre_deberta_0130_list.jsonl
qwen_1p8_instag.jsonl
mixture_raw_reweight_task_type.json
task_type_bge.jsonl
socre_rerank_0201_list.jsonl
task_type_bge_math_score.jsonl

用label_gen目录下的vllm_qwen1p8.py生成qwen_1p8_instag.jsonl；
用label_gen目录下的score_deberta.py生成socre_deberta_0130_list.jsonl；
用label_gen目录下的main_quality.py生成质量打分mixture_raw_reweight_task_type.json；
用label_gen目录下的score_rerank_correlation.py生成socre_rerank_0201_list.jsonl；
用label_gen目录下的run_labeling_task_type.sh脚本以mixture_raw_reweight.jsonl为输入，生成task_type_bge.jsonl文件；
用label_gen目录下的run_labeling_task_type.sh脚本以mixture_raw_math_only_juicer.jsonl为输入，生成task_type_bge_math_score.jsonl文件；

以上这些标签文件全部都已经生成好，放置于solution目录下，无需重复生成。

