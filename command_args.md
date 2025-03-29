lm_eval --model hf --model_args pretrained=/workspace/RWKV-block/test/v7_goose/.hf_build/v7-1B4/,trust_remote_code=True,tmix_backend="fla" --tasks RULER --batch_size 32 --max_length 32768 --apply_chat_template --output_path results.json

lm_eval --model hf --model_args pretrained=/root/v7-1B4/,max_length=32768,trust_remote_code=True,tmix_backend="fla" --tasks niah_single_1 --batch_size 32 --apply_chat_template --output_path results.json

lm_eval --model hf --model_args pretrained='fla-hub/rwkv7-1.5B-world',max_length=16287,trust_remote_code=True,tmix_backend="fla" --tasks niah_single_1 --batch_size 32 --apply_chat_template --output_path results.json

lm_eval --model hf --model_args pretrained=/workspace/RWKV-block/test/v7_goose/.hf_build/v7-2B9-world/,trust_remote_code=True,tmix_backend="fla" --tasks mmlu --batch_size 16 --apply_chat_template


lm_eval --model hf --model_args pretrained='fla-hub/rwkv7-1.5B-world',trust_remote_code=True,max_length=32768 --tasks ruler --batch_size 4 --apply_chat_template

accelerate launch -m lm_eval --model hf --model_args pretrained='fla-hub/rwkv7-2.9B-world',trust_remote_code=True,max_length=32768 --tasks ruler --batch_size 16 --apply_chat_template

accelerate launch -m lm_eval --model hf --model_args pretrained=fla-hub/rwkv7-2.9B-world,trust_remote_code=True --tasks arc_challenge,arc_easy,hellaswag,lambada_openai,piqa,glue,sciq,winogrande,mmlu --batch_size 8 --log_samples --output_path /workspace/lm-evaluation-harness/results



accelerate launch -m lm_eval --model hf --model_args pretrained=fla-hub/rwkv7-2.9B-world,trust_remote_code=True --tasks arc_challenge,arc_easy,hellaswag,lambada_openai,piqa,glue,sciq,winogrande,mmlu --batch_size 8 --output_path /workspace/lm-evaluation-harness/results --hf_hub_log_args hub_results_org=jannalulu/rwkv7-results, push_results_to_hub=True,push_samples_to_hub=True,public_repo=False

accelerate launch -m lm_eval --model hf --model_args pretrained=fla-hub/rwkv7-2.9B-world,trust_remote_code=True --tasks mmlu --num_fewshot 5 --batch_size 8 --log_samples --output_path /workspace/lm-evaluation-harness/results

lm_eval --model hf --model_args pretrained=fla-hub/rwkv7-1.5B-world,trust_remote_code=True --tasks lambada_openai --batch_size 8 --output_path /workspace/lm-evaluation-harness/results --hf_hub_log_args hub_results_org=jannalulu,hub_repo_name=rwkv7-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False

lm_eval --model hf --model_args pretrained=fla-hub/rwkv7-1.5B-world,trust_remote_code=True,add_bos_token=True,dtype=float32 --tasks lambada_openai --batch_size 8 --output_path /workspace/lm-evaluation-harness/results --hf_hub_log_args hub_results_org=jannalulu,hub_repo_name=rwkv7-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False

lm_eval --model hf --model_args pretrained=/workspace/RWKV-block/test/v7_goose/.hf_build/v7-1B5-world/,trust_remote_code=True,add_bos_token=True,tmix_backend="pytorch" --tasks lambada_openai --batch_size 8 --output_path /workspace/lm-evaluation-harness/results



lm_eval --model hf --model_args pretrained=fla-hub/rwkv7-168M-pile,trust_remote_code=True,dtype=float32,add_bos_token=True --tasks lambada_openai --batch_size 16 --output_path /workspace/lm-evaluation-harness/results

lm_eval --model hf --model_args pretrained=SmerkyG/RWKV7-Goose-0.1B-Pile-HF,trust_remote_code=True,dtype=float32,add_bos_token=True --tasks lambada_openai --batch_size 16 --output_path /workspace/lm-evaluation-harness/results

lm_eval --model hf --model_args pretrained=fla-hub/rwkv7-1.5B-world,trust_remote_code=True,dtype=float32,max_length 16384,chunked_generation_size 2048 --tasks niah_single_1 --batch_size 8 --output_path /workspace/lm-evaluation-harness/results

accelerate launch -m lm_eval --model mamba_ssm --model_args pretrained=state-spaces/mamba2-2.7b,dtype=float32 --device cuda --tasks arc_challenge,arc_easy,hellaswag,lambada_openai,piqa,glue,sciq,winogrande --batch_size 64 --output_path /workspace/lm-evaluation-harness-0.4.3/results

accelerate launch -m lm_eval --model mamba_ssm --model_args pretrained=state-spaces/mamba-1.4b,dtype=float32 --device cuda --tasks mmlu --batch_size 64 --output_path /workspace/lm-evaluation-harness-0.4.3/results

accelerate launch -m lm_eval --model --model_args pretrained=/workspace/lm-evaluation-harness-0.4.3/RWKV-x070-Pile-1.47B-20241210-ctx4096.pth,dtype=float32 --device cuda --tasks arc_challenge,arc_easy,hellaswag,lambada_openai,piqa,glue,sciq,winogrande --batch_size 64 --output_path /workspace/lm-evaluation-harness-0.4.3/results

accelerate launch -m lm_eval --model hf --model_args pretrained=google/gemma-2-9b,trust_remote_code=True,add_bos_token=True --device cuda --tasks arc_challenge,arc_easy,hellaswag,lambada_openai,piqa,glue,sciq,winogrande,mmlu --batch_size 12 --output_path /workspace/lm-evaluation-harness/results

accelerate launch -m lm_eval --model hf --model_args pretrained=SmerkyG/rwkv7-0.4B-world,trust_remote_code=True --tasks arc_challenge,arc_easy,hellaswag,lambada_openai,piqa,glue,sciq,winogrande,mmlu --batch_size 128 --output_path /workspace/lm-evaluation-harness/results

accelerate launch -m lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-7B,trust_remote_code=True --device cuda --tasks arc_challenge,arc_easy,hellaswag,lambada_openai,piqa,winogrande,mmlu --batch_size 8 --output_path /workspace/lm-evaluation-harness/results

accelerate launch -m lm_eval --model hf --model_args pretrained=RWKV-Red-Team/ARWKV-7B-Preview-0.1,trust_remote_code=True,dtype=float16 --device cuda --tasks arc_easy,arc_challenge,hellaswag,lambada_openai,piqa,winogrande,mmlu --batch_size 64 --output_path /workspace/lm-evaluation-harness/results-0.4.8

python lm_harness_eval.py --model hybrid-phi-mamba --tasks lambada_openai,piqa,winogrande,mmlu --device cuda --batch_size 64 --output_path /workspace/lm-evaluation-harness/results-0.4.8

accelerate launch -m lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.1-8B,trust_remote_code=True --device cuda --tasks mmlu --batch_size 7 --num_fewshot 5 --output_path /workspace/lm-evaluation-harness/results

accelerate launch -m lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-7B,trust_remote_code=True,dtype=float32 --device cuda --tasks lambada_multilingual,pawsx,xcopa,xnli,xstorycloze,xwinograd --batch_size 8 --output_path /workspace/lm-evaluation-harness-0.4.3/results

accelerate launch -m lm_eval --model hf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,trust_remote_code=True,dtype=float32 --device cuda --tasks arc_easy,arc_challenge,lambada_openai,piqa,winogrande,mmlu,hellaswag --batch_size 8 --output_path /workspace/lm-evaluation-harness/results-0.4.8

accelerate launch -m lm_eval --model hf --model_args pretrained=TRI-ML/mistral-supra,trust_remote_code=True,dtype=float16 --device cuda --tasks lambada_openai,piqa,winogrande --batch_size 16 --output_path /workspace/lm-evaluation-harness/results-0.4.8

accelerate launch supra_script.py --model hf --model_args pretrained=TRI-ML/mistral-supra,trust_remote_code=True,dtype=float16 --device cuda --tasks mmlu --batch_size 16 --output_path /workspace/lm-evaluation-harness/results-0.4.8

accelerate launch -m lm_eval --model hf --model_args pretrained=JunxiongWang/Mamba2InLlama_1,trust_remote_code=True,dtype=float16 --device cuda --tasks lambada_openai,piqa,winogrande,mmlu --batch_size 10 --output_path /workspace/lm-evaluation-harness/results-0.4.8

python benchmark/llm_eval/lm_harness_eval.py --model mamba2_hybrid --model_args pretrained=JunxiongWang/Mamba2InLlama_1 --tasks lambada_openai,piqa,winogrande --device cuda --batch_size 16 --output_path /workspace/lm-evaluation-harness/results-0.4.8

lm_eval --model hf --model_args pretrained=EleutherAI/pythia-70m,trust_remote_code=True,dtype=float16 --device cuda --tasks piqa,arc_challenge --batch_size 32 --output_path /workspace/lm-evaluation-harness/results-0.4.8

accelerate launch -m lm_eval --model hf --model_args pretrained=meta-llama/Llama-2-7b-hf,trust_remote_code=True,dtype=float16 --device cuda --tasks lambada_openai,piqa,winogrande,mmlu,arc_easy,arc_challenge,hellaswag --batch_size 8 --output_path /workspace/lm-evaluation-harness/results-0.4.8

export PYTHONPATH=$PYTHONPATH:/workspace/MambaInLlama
python benchmark/llm_eval/lm_harness_eval.py --model mamba2_hybrid --model_args pretrained=JunxiongWang/Mamba2InLlama_0_75 --tasks lambada_openai,piqa --device cuda --batch_size 16 --output_path /workspace/lm-evaluation-harness/results-0.4.8

python -c "from datasets import load_dataset; 
tasks = {
  'cola': 'validation', 
  'mnli_matched': 'validation', 
  'mnli_mismatched': 'validation', 
  'mrpc': 'validation', 
  'qnli': 'validation', 
  'qqp': 'validation', 
  'rte': 'validation', 
  'sst2': 'validation',
  'wnli': 'validation'
};
for task, split in tasks.items():
  try:
    count = len(load_dataset('glue', task, split=split))
    print(f'{task}: {count}')
  except Exception as e:
    print(f'Error loading {task}: {e}')
"