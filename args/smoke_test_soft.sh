#!/bin/bash
#SBATCH --job-name=smoke_test_args_soft
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/net/holy-nfsisilon/ifs/rc_labs/tambe_lab/Everyone/Stackelberg_Reward_Shaping/args/slurm_logs/%x_%j.out
#SBATCH --error=/net/holy-nfsisilon/ifs/rc_labs/tambe_lab/Everyone/Stackelberg_Reward_Shaping/args/slurm_logs/%x_%j.err
#SBATCH --open-mode=append

source ~/.bashrc
conda activate args

export HF_HOME=/n/tambe_lab_tier1/Lab/haichuan/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR="/net/holy-nfsisilon/ifs/rc_labs/tambe_lab/Everyone/Stackelberg_Reward_Shaping/args"
LLM_PATH="/n/tambe_lab_tier1/Everyone/Principal-Agent-Alignment/models/qwen3-8b"
RM_PATH="/n/tambe_lab_tier1/Everyone/Principal-Agent-Alignment/models/qwen3-8b-rm"
DATASET_LOCAL_DIR="/n/tambe_lab_tier1/Everyone/Principal-Agent-Alignment/datasets/HH-RLHF"

OUT_FILE="${WORKDIR}/outputs/smoke_test/soft_B_12.0_alpha_1.5_resp10"
mkdir -p "$(dirname "$OUT_FILE")"

echo "[INFO] Running smoke test: 5 prompts only"
echo "[INFO] LLM: $LLM_PATH"
echo "[INFO] RM:  $RM_PATH"

python "${WORKDIR}/new_collect_args_out_soft.py" \
    --dataset "Dahoas/full-hh-rlhf" \
    --dataset_local_dir "$DATASET_LOCAL_DIR" \
    --setting test \
    --llm "$LLM_PATH" \
    --rm "$RM_PATH" \
    --llm_gpu cuda:0 \
    --rm_gpu cuda:1 \
    --max_new_token 64 \
    --config "${WORKDIR}/configs/args_soft_hh.jsonl" \
    --out_file "$OUT_FILE" \
    --B 12.0 \
    --alpha 1.5 \
    --K 5 \
    --recover

echo "[INFO] Smoke test done. Output:"
cat "${OUT_FILE}.jsonl" 2>/dev/null || echo "[WARN] No output file found."
