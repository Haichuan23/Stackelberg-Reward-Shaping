# Stackelberg Reward Shaping

<p align="center">
  <br />
  <a href="./LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-red.svg" /></a>
  <a href="Python 3.8"><img alt="Python 3.8" src="https://img.shields.io/badge/python-3.8-blue.svg" /></a>
</p>

<!-- ![A brief overview of the Composite Flow Framework.](./imgs/compositeflow.png) -->

🔥 [Reward Shaping for Inferernce-Time Alignment: A Stackelberg Game Perspective](https://arxiv.org/abs/2602.02572)

Stackelberg Reward Shaping studies optimal reward shaping for LLM alignment by formulating the alignment process as a Stackelberg game, or more specifically, a principal–agent model. The reward-model provider acts as the leader, designing a reward function as an incentive mechanism, while the language model acts as the follower and best responds by selecting an aligned policy. 

Our codebase supports reward shaping on two popular inference-time alignment methods: 
1. Alignment as Reward Guided Search (ARGS)
2. Controlled Decoding (CD)

## Installation (SRS)
To download the required packages, run 
```bash
pip install -r requirements.txt
```

To download models or eval dataset, run 
```bash
huggingface-cli download Qwen/Qwen3-8B \
  --local-dir "models/qwen3-8b" \
  --resume-download

huggingface-cli download Skywork/Skywork-Reward-V2-Qwen3-8B \
  --local-dir "models/qwen3-skywork" \
  --resume-download
```

## Run Stackelberg Reward Shaping on ARGS
Go to the args subfolder. To run our reward shaping method, you can run with
```bash
python new_collect_args_out_soft.py \
    --dataset Dahoas/full-hh-rlhf \
    --dataset_local_dir datasets/HH-RLHF \
    --setting test \
    --llm ./models/qwen3-8b \
    --rm ./models/qwen3-skywork \
    --llm_gpu cuda:0 \
    --rm_gpu cuda:1 \
    --max_new_token 128 \
    --config configs/args_soft_hh.jsonl \
    --out_file outputs/qwen3-8b-hh/skywork/greedy/Lambda_0.5/soft_B_12.0_alpha_1.5_resp10 \
    --B 12.0 \
    --alpha 1.5 \
    --recover
```

Soft here refers to Stackelberg Reward Shaping (soft) in the paper. The baselines can be run similary. 

## Run Stackelberg Reward Shaping on CD
Go to the cd subfolder. To run cd, one need to construct an offline dataset first using the following code:
```bash
python util_offline/generate_offline_data_stop.py \
  --model_name_or_path ./models/qwen3-8b \
  --reward_model_name_or_path ./models/qwen3-skywork \
  --rm_type standard \
  "${LM_DTYPE_ARG[@]}" \
  --split "train" \
  --num_problems "10000" \
  --samples_per_prompt "10" \
  --max_new_tokens "128" \
  --temperature "0.7" \
  --top_p "0.9" \
  --seed "1234" \
  --output_dir "datasets/" \
  --dataset_name "HH-RLHF" \
  --dataset_root "datasets/" \
  --fp16_hidden \
  --write_jsonl \
  --stop_on_human \
  --stop_sentinels "\n\nHuman:" "\nHuman:" "\n\nUser:" "\nUser:" \
  --start_prompt "0" \
  --end_prompt "1000" \
  --lm_gpu_id "0" \
  --rm_gpu_id "1"
```

To perform Stackelberg Reward Shaping offline, run the following code:
Go to the cd subfolder. To run cd, one need to construct an offline dataset first using the following code:
```bash
python -u util_offline/compute_offline_shaped_reward_soft.py \
  --root_dir datasets/models_qwen3-8b-hh \
  --reward_model skywork \
  --max_prompts 10000 \
  --num_responses 10 \
  --B 15 \
  --beta 0.667 \
  --cap 2.0 \
  --alpha 1.0
```

To train the $Q_{\phi}^{\mathrm{SRS}}$, run the following code:
```bash
python util_offline/train_value_function.py \
  --shards_dir datasets/models_qwen3-8b-hh/shards \
  --reward_model skywork \
  --response_num 10 \
  --B 15 \
  --beta 0.667 \
  --alpha 1.0 \
  --reward_type soft \
  --max_prompt 10000 \
  --max_len 128 \
  --batch_size 64 \
  --epochs 15 \
  --lr 1e-4 \
  --device cuda \
  --num_workers 4 \
  --pin_memory \
  --seed 42 \
  --val_split 0.05 \
  --use_pairwise \
  --pairwise_coef 1.0 \
  --ckpt_dir checkpoints/qwen3-8b-hh/skywork/soft_value_fn_resp10_B15_beta0.667_alpha1.0_seed42_prompt10000_cap2.0 \
  --model_name qwen3-8b \
  --patience 5 \
  --evaluation hh \
  --cap 2.0
```

To use the trained Q function for decoding, run the following code:
```bash
python collect_cd_outs.py \
  --dataset Dahoas/full-hh-rlhf \
  --dataset_local_dir datasets/HH-RLHF \
  --model_name qwen3-8b \
  --reward_model skywork \
  --setting test \
  --out_file greedy-10000LAMBDA_1.5/soft_resp10_B15_beta0.667_alpha1.0_seed42_prompt_10000/ctrl_collect_greedy_lambda_1.5.jsonl \
  --value_ckpt checkpoints/qwen3-8b-hh/skywork/soft_value_fn_resp10_B15_beta0.667_alpha1.0_seed42_prompt10000_cap2.0/best_value_agent.pt \
  --model_id ./models/qwen3-8b \
  --device cuda:0 \
  --dtype float16 \
  --lambda_coef 1.5 \
  --top_k 10 \
  --temperature 0.7 \
  --mode greedy \
  --num_beams 4 \
  --sample_temperature 1.0 \
  --max_new_tokens 128 \
  --evaluation hh
```

## Evaluation



## 📄Citing Stackelberg Reward Shaping
Please consider citing us if you find our work useful!

```
@inproceedings{wang2026reward,
  title={Reward Shaping for Inference-Time Alignment: A Stackelberg Game Perspective},
  author={Wang, Haichuan and Lin, Tao and Kong, Lingkai and Li, Ce and Jiang, Hezi and Tambe, Milind},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  year={2026}
}
```