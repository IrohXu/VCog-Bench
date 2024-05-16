# VCog: Can Multimodal LLMs Learn Visual Cognition?

### Background


### Build


### VCog Benchmark

```
python mllm_inference.py --llm "gpt4v" --model-weight 'gpt-4-vision-preview' --root '/home/xucao2/VLM_experiment/VCog/dataset/task1/tf1/md' --choice "full-image"
```

```
python mllm_inference.py --llm "gpt4" --model-weight 'gpt-4-turbo' --root '/home/xucao2/VLM_experiment/VCog/dataset/task1/tf3/pd' --choice "context"
```

```
python mllm_inference.py --llm "claude3v" --model-weight 'claude-3-opus-20240229' --root '/home/xucao2/VLM_experiment/VCog/dataset/task1/tf1/md' --choice "full-image"
```

```
python mllm_inference.py --llm "claude3" --model-weight 'claude-3-opus-20240229' --root '/home/xucao2/VLM_experiment/VCog/dataset/task1/tf3/pd' --choice "context"
```

```
python mllm_inference.py --llm "geminiv" --model-weight 'gemini-1.5-pro-latest' --root '/home/xucao2/VLM_experiment/VCog/dataset/task1/tf1/md' --choice "full-image"
```

```
python mllm_evaluate.py --llm 'claude3' --main-path '/home/xucao2/VLM_experiment/VCog/dataset/task1/tf2/md'
```

```
python mllm_evaluate.py --llm 'claude3' --main-path '/home/xucao2/VLM_experiment/VCog/dataset/task1/tf3/md'
```

```
python mllm_evaluate.py --llm 'claude3' --main-path '/home/xucao2/VLM_experiment/VCog/dataset/task1/tf3/pd'
```

### Methods




