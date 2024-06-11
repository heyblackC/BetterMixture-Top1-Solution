# 需要确保网络能够访问huggingface
# 判断文件是否存在

################ 一、混合数据 ####################
local_file="./tools/math_score.bin"
if [ ! -f "$local_file" ]; then
    wget https://huggingface.co/open-web-math/filtering-models/resolve/main/math_score.bin
    mv math_score.bin ./tools
else
    echo "File already exists."
fi

# 1. tools/tool_inputcheck.py依赖了open-web-math项目公开的math_score.bin数学打分模型，因此需要从huggingface网站下载模型。
# 2. 观察到COIG_translate_zh/en的数据质量太低，因此从源头上就删除这两个数据集,当前方法的重点在数据过滤和采样方案。
python get_mixture.py