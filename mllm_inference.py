import argparse
from PIL import Image

from langchain_core.messages import HumanMessage, SystemMessage
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

parser.add_argument('--llm', type=str, default='gpt4v')
parser.add_argument('--model-weight', type=str, default='gpt-4-vision-preview')
parser.add_argument('--root', type=str)
parser.add_argument('--choice', type=str, default='full-image')

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

if __name__ == "__main__":
    args = parser.parse_args()
    
    main_path = args.root
    choice = args.choice
    
    if args.llm == "gpt4" or args.llm == "gpt4v":
        chat = ChatOpenAI(model=args.model_weight, max_tokens=384)
    elif args.llm == "claude3" or args.llm == "claude3v":
        chat = ChatAnthropic(model=args.model_weight, max_tokens=384)
    elif args.llm == "gemini" or args.llm == "geminiv":
        chat = ChatGoogleGenerativeAI(model=args.model_weight, max_tokens=384)
    else:
        raise("Model does not exist.")
    
    for case_path in os.listdir(main_path):
        print(case_path)
        example_path = os.path.join(main_path, case_path)
        
        answer_img_path = os.path.join(main_path, case_path, "answer", "image")
        if choice == "full-image":
            choice_img_path = os.path.join(main_path, case_path, "choiceX", "image")
        else:
            choice_img_path = os.path.join(main_path, case_path, "choice", "image")
        
        if choice == "context":
            choice_text_path = os.path.join(main_path, case_path, "choice", "text", "annotation.json")
            choice_json = json.load(open(choice_text_path))
        
        question_img_path = os.path.join(main_path, case_path, "question", "image")
        
        result_path = os.path.join(example_path, "result")
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        
        save_path = os.path.join(result_path, args.llm + ".json")
        
        if os.path.exists(save_path):
            continue
        
        answer_image_name = os.listdir(answer_img_path)[0]
        choices_image_names = os.listdir(choice_img_path)
        question_path_name = os.listdir(question_img_path)[0]
        
        encoded_question_img = encode_image(os.path.join(question_img_path, question_path_name))
        
        encoded_choice_imgs = []
        choice_imgs = []
        for img_path in choices_image_names:
            choice_imgs.append(img_path)

            temp_path = os.path.join(choice_img_path, img_path)
            encoded_choice_imgs.append(encode_image(temp_path))
        
        system_template = load_prompt(f"./vcog_prompt_context.txt")
        response_format = load_prompt(f"./vcog_response_format.txt")
        
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        
        system_message = system_message_prompt.format(
            response_format=response_format,
        )
        input_list = [system_message]
        
        input_list.append(add_image_input(encoded_question_img, "Question:"))
        
        if choice == "full-image" or choice == "image":
            for i in range(0, len(encoded_choice_imgs)):
                choice_img = encoded_choice_imgs[i]
                input_list.append(add_image_input(choice_img, "Choice " + str(i+1)))
        elif choice == "context":
            for i in range(0, len(encoded_choice_imgs)):
                option_context = choice_json[choices_image_names[i]]
                input_list.append(add_context_input("Choice " + str(i+1) + ": " + option_context + "\n"))
        else:
            print("skip")
            continue
            
        output = chat.invoke(input_list)
        
        print(output.content)
        
        output_json = json.loads(output.content)
        output_json["Answer"] = choices_image_names[output_json["Answer"]-1]
        output_json["Groundtruth"] = answer_image_name
        
        output_json = json.dumps(output_json, indent=3)
        
        with open(save_path, "w") as out_file:
            out_file.write(output_json)
