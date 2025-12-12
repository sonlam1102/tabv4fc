CUDA_VISIBLE_DEVICES=1 python llms.py  \
    --model Qwen/Qwen2.5-7B-Instruct \
    --data_name pubhealthtab \
    --data_path ./results/summary/pubhealth_tab_FC.json