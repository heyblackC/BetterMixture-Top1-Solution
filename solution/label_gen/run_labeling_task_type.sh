export WANDB_MODE=disabled

INPUT_DIR=./input_dir
OUTPUT_DIR=./output_dir

for i in {0..0}
do
    CUDA_VISIBLE_DEVICES=$i nohup python topic_labeling_task_type.py --cleaned_input_dir $INPUT_DIR --output_dir $OUTPUT_DIR > logs/labeling_task_gpu$i.out 2>&1 &
    sleep 5
done
