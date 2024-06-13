import argparse
import os
import random
from collections import defaultdict

import cv2
import re

import numpy as np
from PIL import Image
import torch
import html
import gradio as gr

import torchvision.transforms as T
import torch.backends.cudnn as cudnn

from minigpt4.common.config import Config

from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

import json

parser = argparse.ArgumentParser()

parser.add_argument('--llm', type=str, default='minigpt')
parser.add_argument("--cfg-path", default='minigpt4/eval_configs/minigptv2_eval.yaml',
                        help="path to configuration file.")
parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
            "in xxx=yyy format will be merged into config file (deprecate), "
            "change to --cfg-options instead.",
)
parser.add_argument('--max-tokens', type=int, default=1024)
parser.add_argument('--dataset-path', type=str)
parser.add_argument('--system-template', type=str, default='./minigpt_prompt_raven.txt')


def load_prompt(prompt_path):
    with open(prompt_path, 'r') as f:
        prompt = f.read()
    return prompt


def prepare_texts(texts, conv_temp):
    convs = [conv_temp.copy() for _ in range(len(texts))]
    [conv.append_message(
        conv.roles[0], '<Img><ImageHere></Img> {}'.format(text)) for conv, text in zip(convs, texts)]
    [conv.append_message(conv.roles[1], None) for conv in convs]
    texts = [conv.get_prompt() for conv in convs]
    return texts


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    cudnn.benchmark = False
    cudnn.deterministic = True

    print('Initializing Chat')
    args = parser.parse_args()
    
    main_path = args.dataset_path
    
    cfg = Config(args)
    
    device = 'cuda:{}'.format(args.gpu_id)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    bounding_box_size = 100

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    model = model.eval()

    CONV_VISION = Conversation(
        system="",
        roles=(r"<s>[INST] ", r" [/INST]"),
        messages=[],
        offset=2,
        sep_style=SeparatorStyle.SINGLE,
        sep="",
    )
    

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
        for i in range(0, len(choices_image_names)):
            option_context = choice_json[choices_image_names[i]]
            input_list.append("Option " + str(i+1) + ": " + option_context + "\n")
        
        conv_temp = CONV_VISION.copy()
        texts = [system_template + "\n" + " ".join(input_list)]
        texts = prepare_texts(texts, conv_temp)
        
        raw_image = Image.open(os.path.join(question_img_path, question_path_name)).convert('RGB')
        image = vis_processor(raw_image).unsqueeze(0).to(device)
        answers = model.generate(image, texts, max_new_tokens=args.max_tokens, do_sample=False)

        output_json = {}
        
        output_json["answer"] = choices_image_names[int(answers[0])-1]
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



