#GRPO training related

## First version
For post-train alignment algorithms details about [DPO](https://arxiv.org/pdf/2305.18290) an [GRPO](https://arxiv.org/pdf/2405.20304), please refer to the paper. And for the basic [more RLHF](https://github.com/opendilab/awesome-RLHF)
1. with gsm8k dataset(Grad school math), LORA peft training
2. trained with MI300x 8 cards, the scripts is run_grpo.sh, docker image rocm/pytorch-training:latest
3. Primary training with reference from [unsloth document](https://docs.unsloth.ai/basics/reasoning-grpo-and-rl)
4. Current version is based on pytorch inference (no vllm or SGLANG used)

