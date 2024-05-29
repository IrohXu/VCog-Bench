import argparse
from PIL import Image

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts import SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
import base64
import os
import json
import re

parser = argparse.ArgumentParser()

parser.add_argument('--llm', type=str, default='gpt4')
parser.add_argument('--model-weight', type=str, default='gpt-4-turbo')
parser.add_argument('--max-tokens', type=int, default=384)
parser.add_argument('--dataset-path', type=str)
parser.add_argument('--choice', choices=['image', 'full-image', 'context'])
parser.add_argument('--system-template', type=str, default='./vcog_prompt_image.txt')
parser.add_argument('--response-format', type=str, default='./vcog_response_format.txt')
parser.add_argument('--few-shot', action='store_true')
parser.add_argument('--shot-num', type=int, default=1)
parser.add_argument('--few-shot-path', type=str, default='./dataset/fewshot')


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def load_prompt(prompt_path):
    with open(prompt_path, 'r') as f:
        prompt = f.read()
    return prompt


def add_image_input(img, context):
    out = HumanMessage(
        content=[
            {"type": "text", "text": context},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img}",
                    "detail": "auto",
                },
            },
        ]
    )
    return out


def add_context_input(context):
    out = HumanMessage(
        content=[
            {"type": "text", "text": context}
        ]
    )
    return out


def add_fewshot_sample(img_question, img_option_set, answer_idx, reason):
    out = []
    out.append(HumanMessage(
        content=[
            {"type": "text", "text": "Question:"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_question}",
                    "detail": "auto",
                },
            },
        ]
    ))
    for i in range(0, len(img_option_set)):
        out.append(HumanMessage(
            content=[
                {"type": "text", "text": "Option " + str(i+1) + ":"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_option_set[i]}",
                        "detail": "auto",
                    },
                },
            ]
        ))
    out.append(AIMessage(
        content=[
            {"type": "text", "text": "The correct answer is Option " + str(answer_idx) + ". " + reason},
        ]
    ))   
    return out


def set_model(llm_name, model_weight, max_tokens):
    if "gpt" in llm_name:
        chat = ChatOpenAI(model=model_weight, max_tokens=max_tokens)
    elif "claude" in llm_name:
        chat = ChatAnthropic(model=model_weight, max_tokens=max_tokens)
    elif "gemini" in llm_name:
        chat = ChatGoogleGenerativeAI(model=model_weight, max_tokens=max_tokens)
    else:
        raise("Model does not exist.")
    return chat


if __name__ == "__main__":
    args = parser.parse_args()
    
    main_path = args.dataset_path
    choice = args.choice
    few_shot = args.few_shot
    main_save_path = os.path.join("./results", args.llm + ".json")
    if os.path.exists(main_save_path):
        sample_save_path_list = json.load(open(main_save_path))
    else:
        sample_save_path_list = {}

    chat = set_model(llm_name=args.llm, model_weight=args.model_weight, max_tokens=args.max_tokens)
        
    for case_path in os.listdir(main_path):
        print(case_path)
        if case_path in sample_save_path_list:
            continue
        example_path = os.path.join(main_path, case_path)
                                    
        encoded_question_img = encode_image(example_path)
                    
        system_template = load_prompt(args.system_template)
        response_format = load_prompt(args.response_format)
        
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        system_message = system_message_prompt.format(
            response_format=response_format,
        )
        input_list = [system_message]
                
        input_list.append(add_image_input(encoded_question_img, "Question:"))
            
        output = chat.invoke(input_list)
        output = output.content    
        print(output)
        match = re.search(r'\{.*\}', output, flags=re.DOTALL)
        output = match.group()
        
        output_json = json.loads(output)
        output_json["id"] = case_path
        
        output_json["groundtruth"] = 4
        
        sample_save_path_list[case_path] = output_json["answer"]
        sample_output = json.dumps(sample_save_path_list)
        
        with open(main_save_path, "w") as out_file:
            out_file.write(sample_output)
                
