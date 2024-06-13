import argparse
import os
import random
from collections import defaultdict

import re

import numpy as np
from PIL import Image
import torch
import html

import torchvision.transforms as T
import torch.backends.cudnn as cudnn

from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map

import json

parser = argparse.ArgumentParser()

parser.add_argument('--llm', type=str, default='cogvlmv2')
parser.add_argument('--dataset-path', type=str)
parser.add_argument('--system-template', type=str, default='./minigpt_prompt_raven.txt')


MODEL_PATH = ""
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16

def load_prompt(prompt_path):
    with open(prompt_path, 'r') as f:
        prompt = f.read()
    return prompt

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=TORCH_TYPE,
        trust_remote_code=True,
    )

num_gpus = torch.cuda.device_count()
max_memory_per_gpu = "24GiB"
if num_gpus >= 2:
    max_memory_per_gpu = f"{round(42 / num_gpus)}GiB"

device_map = infer_auto_device_map(
    model=model,
    max_memory={i: max_memory_per_gpu for i in range(num_gpus)},
    no_split_module_classes=["CogVLMDecoderLayer"]
)
model = load_checkpoint_and_dispatch(model, MODEL_PATH, device_map=device_map, dtype=TORCH_TYPE, offload_folder="/home/xucao2/VLM_experiment/checkpoints/offload_folder")
model = model.eval()

text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"

if __name__ == "__main__":
    args = parser.parse_args()
    
    main_path = args.dataset_path
    
    sample_save_path_list = []     
    output_list = []   
    main_save_path = os.path.join("./results", args.llm + ".json")
    
    for case_path in os.listdir(main_path):
        print(case_path)
        example_path = os.path.join(main_path, case_path)
        
        question_img_path = os.path.join(main_path, case_path, "question", "image")
        answer_img_path = os.path.join(main_path, case_path, "answer", "image")
      
        choice_img_path = os.path.join(main_path, case_path, "choice", "image")
        choice_text_path = os.path.join(main_path, case_path, "choice", "text", "annotation.json")
        choice_json = json.load(open(choice_text_path))
            
        result_path = os.path.join(example_path, "result")
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        save_path = os.path.join(result_path, args.llm + ".json")
        
        if os.path.exists(save_path):
            output = load_prompt(save_path)
            output_json = json.loads(output)
            output_list.append(output_json)
            sample_save_path_list.append(save_path)
            continue
        
        answer_image_name = os.listdir(answer_img_path)[0]
        choices_image_names = os.listdir(choice_img_path)
        question_path_name = os.listdir(question_img_path)[0]
                   
        system_template = load_prompt(args.system_template)
        input_list = []
        random.shuffle(choices_image_names)
                
        for i in range(0, len(choices_image_names)):
            option_context = choice_json[choices_image_names[i]]
            input_list.append("Option " + str(i+1) + ": " + option_context + "\n")
        
        
        raw_image = Image.open(os.path.join(question_img_path, question_path_name)).convert('RGB')

        query = [system_template + "\n" + " ".join(input_list)]

        input_by_model = model.build_conversation_input_ids(
            tokenizer,
            query=query,
            history=[],
            images=[raw_image],
            template_version='chat'
        )
        
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
            'images': [[input_by_model['images'][0].to(DEVICE).to(TORCH_TYPE)]] if raw_image is not None else None,
        }
        gen_kwargs = {
            "max_new_tokens": 2048,
            "pad_token_id": 128002,  
        }
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0])
            response = response.split("<|end_of_text|>")[0]
            print(query, "\nCogVLM2:", response)

    
        output_json = {}
        
        match = re.search(r'Option \d', response, flags=re.DOTALL)
        output = match.group()
        
        match = re.search(r'\d', output, flags=re.DOTALL)
        output = match.group()
        
        output_json["answer"] = choices_image_names[int(output)-1]
        output_json["groundtruth"] = answer_image_name
        
        output_list.append(output_json)
        
        sample_output = json.dumps(output_json, indent=2)
                
        with open(save_path, "w") as out_file:
            out_file.write(sample_output)
        
        sample_save_path_list.append(save_path)
        
        main_output = json.dumps(output_list)
    
        with open(main_save_path, "w") as out_file:
            out_file.write(main_output)
        
        print(output_json)

    main_output = json.dumps(output_list)
    
    with open(main_save_path, "w") as out_file:
        out_file.write(main_output)
    
    for save_path in sample_save_path_list:
        os.remove(save_path)


