# VCog-Bench: Benchmarking Multimodal LLMs on Abstract Visual Reasoning

### Background
Recently, Multimodal Large Language Models (MLLMs) have shown great promise in language-guided perceptual tasks such as recognition, segmentation, and object detection. However, their effectiveness in addressing visual cognition problems that require high-level reasoning is not well-established. One such challenge is abstract visual reasoning (AVR) -- the cognitive ability to discern relationships among patterns in a set of images and extrapolate to predict subsequent patterns. This skill is crucial during the early neurodevelopmental stages of children. Inspired by the AVR tasks in Raven’s Progressive Matrices (RPM) and Wechsler Intelligence Scale for Children (WISC), we propose a new dataset MaRs-VQA and a new benchmark VCog-Bench containing three datasets to evaluate the zero-shot AVR capability of MLLMs and compare their performance with existing human intelligent investigation. Our comparative experiments with different open-source and closed-source MLLMs on the VCog-Bench revealed a gap between MLLMs and human intelligence, highlighting the visual cognitive limitations of current MLLMs. We believe that the public release of VCog-Bench, consisting of MaRs-VQA, and the inference pipeline will drive progress toward the next generation of MLLMs with human-like visual cognition abilities.     

### Build

```
pip install -r requirements.txt 
```

### VCog Benchmark

Export Keys from closed-sourse models.    

```
export OPENAI_API_KEY=xxx
export ANTHROPIC_API_KEY=xxx
export GOOGLE_API_KEY=xxx
```

Test MLLMs with multi-images input in the MaRs-VQA dataset in VCog-Bench:    

```
python mllm_inference.py --llm "gpt4o-cot" \
            --model-weight 'gpt-4o' \
            --max-tokens 1024 \
            --dataset-path '' \
            --system-template './prompt/vcog_prompt_image_marsvqa_cot.txt' \
            --response-format './prompt/vcog_response_format_marsvqa_cot.txt' \
            --choice 'image'
```

Test MLLMs with question image - language options input in the MaRs-VQA dataset in VCog-Bench:    

```
python mllm_inference.py --llm "gpt4o-cot" \
            --model-weight 'gpt-4o' \
            --max-tokens 1024 \
            --dataset-path '' \
            --system-template './prompt/vcog_prompt_image_marsvqa_cot.txt' \
            --response-format './prompt/vcog_response_format_marsvqa_cot.txt' \
            --choice 'context'
```



