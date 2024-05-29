# VCog: Can Multimodal LLMs Learn Visual Cognition?

### Background


### Build


### VCog Benchmark


```
python mllm_inference.py --llm "gpt4v-cot-raven" \
            --model-weight 'gpt-4-turbo' \
            --max-tokens 384 \
            --dataset-path '/home/xucao2/VLM_experiment/VCog/testset/raven' \
            --system-template '/home/xucao2/VLM_experiment/VCog/vcog_prompt_image_raven_cot.txt' \
            --response-format '/home/xucao2/VLM_experiment/VCog/vcog_response_format_raven_cot.txt' \
            --choice 'image'
```

```
python mllm_inference.py --llm "gpt4v-0shot-raven-text" \
            --model-weight 'gpt-4-turbo' \
            --max-tokens 1024 \
            --dataset-path '/home/xucao2/VLM_experiment/VCog/testset/raven' \
            --system-template '/home/xucao2/VLM_experiment/VCog/vcog_prompt_image_raven.txt' \
            --response-format '/home/xucao2/VLM_experiment/VCog/vcog_response_format_raven.txt' \
            --choice 'context'
```

```
python mllm_inference.py --llm "claude-3-opus-cot-raven" \
            --model-weight 'claude-3-opus-20240229' \
            --max-tokens 1024 \
            --dataset-path '/home/xucao2/VLM_experiment/VCog/testset/raven' \
            --system-template '/home/xucao2/VLM_experiment/VCog/vcog_prompt_image_raven_cot.txt' \
            --response-format '/home/xucao2/VLM_experiment/VCog/vcog_response_format_raven_cot.txt' \
            --choice 'image'
```

```
python mllm_evaluate.py --json-path '/home/xucao2/VLM_experiment/VCog/results/gpt4o-0shot-marsvqa-text.json'
```

```
python mllm_inference.py --llm "claude-3-opus-0shot-marsvqa-text" \
            --model-weight 'claude-3-opus-20240229' \
            --max-tokens 512 \
            --dataset-path '/home/xucao2/VLM_experiment/VCog/testset/marsvqa' \
            --system-template '/home/xucao2/VLM_experiment/VCog/prompt/vcog_prompt_image_marsvqa.txt' \
            --response-format '/home/xucao2/VLM_experiment/VCog/prompt/vcog_response_format_marsvqa.txt' \
            --choice 'context'
```

```
python mllm_inference.py --llm "claude-3-opus-0shot-marsvqa" \
            --model-weight 'claude-3-opus-20240229' \
            --max-tokens 512 \
            --dataset-path '/home/xucao2/VLM_experiment/VCog/testset/marsvqa' \
            --system-template '/home/xucao2/VLM_experiment/VCog/prompt/vcog_prompt_image_marsvqa.txt' \
            --response-format '/home/xucao2/VLM_experiment/VCog/prompt/vcog_response_format_marsvqa.txt' \
            --choice 'image'
```

```
python mllm_inference_cvr.py --llm "claude-3-haiku-0shot-cvr" \
            --model-weight 'claude-3-haiku-20240307' \
            --max-tokens 512 \
            --dataset-path '/home/xucao2/VLM_experiment/VCog/testset/cvr' \
            --system-template '/home/xucao2/VLM_experiment/VCog/prompt/vcog_prompt_image_cvr.txt' \
            --response-format '/home/xucao2/VLM_experiment/VCog/prompt/vcog_response_format_cvr.txt' \
            --choice 'image'
```

```
python mllm_inference_cvr.py --llm "gpt4o-0shot-cvr" \
            --model-weight 'gpt-4o' \
            --max-tokens 512 \
            --dataset-path '/home/xucao2/VLM_experiment/VCog/testset/cvr' \
            --system-template '/home/xucao2/VLM_experiment/VCog/prompt/vcog_prompt_image_cvr.txt' \
            --response-format '/home/xucao2/VLM_experiment/VCog/prompt/vcog_response_format_cvr.txt' \
            --choice 'image'
```




```
python mllm_inference.py --llm "gpt4-0shot" \
            --model-weight 'gpt-4o' \
            --max-tokens 256 \
            --dataset-path '/home/xucao2/VLM_experiment/VCog/dataset/task1/tf1/md' \
            --choice 'image'
```

```
python mllm_inference.py --llm "gpt4-fewshot" \
            --model-weight 'gpt-4-turbo' \
            --max-tokens 1024 \
            --dataset-path '/home/xucao2/VLM_experiment/VCog/dataset/example' \
            --choice 'image' \
            --few-shot \
            --shot-num 3 \
            --few-shot-path './dataset/fewshot'
```

```
python mllm_evaluate.py --json-path '/home/xucao2/VLM_experiment/VCog/gpt4-0shot-raven-center_single.json'
```

### Methods




