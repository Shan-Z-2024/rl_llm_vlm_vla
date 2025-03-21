accelerate launch --multi_gpu train_grpo.py \
       --model_name="Qwen/Qwen2.5-1.5B-Instruct" \
       --train_epoch=1 \
       --report_to="wandb"
