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
    output_list = []
    main_save_path = os.path.join("./results", args.llm + ".json")
    sample_save_path_list = []
    
    chat = set_model(llm_name=args.llm, model_weight=args.model_weight, max_tokens=args.max_tokens)
    
    few_shot_info = []
    if few_shot:
        fs_main_path = args.few_shot_path
        few_shot_num = args.shot_num
        if not os.path.exists(fs_main_path):
            raise("Few-shot dataset path does not exist.")
        
        for k in range(len(os.listdir(fs_main_path))):
            if k >= few_shot_num:
                break
            fs_case_path = os.listdir(fs_main_path)[k]
            if choice == "full-image":
                fs_choice_img_path = os.path.join(fs_main_path, fs_case_path, "choiceX", "image")
            else:
                fs_choice_img_path = os.path.join(fs_main_path, fs_case_path, "choice", "image")
            fs_answer_img_path = os.path.join(fs_main_path, fs_case_path, "answer", "image")
            fs_reason_path = os.path.join(fs_main_path, fs_case_path, "answer", "reason.txt")
            fs_question_img_path = os.path.join(fs_main_path, fs_case_path, "question", "image")
            
            fs_answer_image_name = os.listdir(fs_answer_img_path)[0]
            fs_choices_image_names = os.listdir(fs_choice_img_path)
            fs_question_path_name = os.listdir(fs_question_img_path)[0]
            fs_encoded_question_img = encode_image(os.path.join(fs_question_img_path, fs_question_path_name))
            reason_context = load_prompt(fs_reason_path)
            fs_encoded_choice_imgs = []
            fs_choice_imgs = []
            answer_index = 0
            for i in range(0, len(fs_choices_image_names)):
                img_path = fs_choices_image_names[i]
                fs_choice_imgs.append(img_path)
                if fs_answer_image_name == img_path:
                    answer_index = i + 1
                temp_path = os.path.join(fs_choice_img_path, img_path)
                fs_encoded_choice_imgs.append(encode_image(temp_path))
        
            few_shot_info += add_fewshot_sample(fs_encoded_question_img, fs_encoded_choice_imgs, answer_index, reason_context)
        
    for case_path in os.listdir(main_path):
        print(case_path)
        example_path = os.path.join(main_path, case_path)
        answer_img_path = os.path.join(main_path, case_path, "answer", "image")
        if choice == "full-image":
            choice_img_path = os.path.join(main_path, case_path, "choiceX", "image")
        elif choice == "image":
            choice_img_path = os.path.join(main_path, case_path, "choice", "image")
        else: # choice == "context":        
            choice_img_path = os.path.join(main_path, case_path, "choice", "image")
            choice_text_path = os.path.join(main_path, case_path, "choice", "text", "annotation.json")
            choice_json = json.load(open(choice_text_path))
            
        question_img_path = os.path.join(main_path, case_path, "question", "image")
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
        
        encoded_question_img = encode_image(os.path.join(question_img_path, question_path_name))
        
        encoded_choice_imgs = []
        choice_imgs = []
        for img_path in choices_image_names:
            choice_imgs.append(img_path)
            temp_path = os.path.join(choice_img_path, img_path)
            encoded_choice_imgs.append(encode_image(temp_path))
            
        system_template = load_prompt(args.system_template)
        response_format = load_prompt(args.response_format)
        
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        system_message = system_message_prompt.format(
            response_format=response_format,
        )
        input_list = [system_message] + few_shot_info
                
        input_list.append(add_image_input(encoded_question_img, "Question:"))
        if choice == "full-image" or choice == "image":
            for i in range(0, len(encoded_choice_imgs)):
                choice_img = encoded_choice_imgs[i]
                input_list.append(add_image_input(choice_img, "Option " + str(i+1)))
        elif choice == "context":
            for i in range(0, len(encoded_choice_imgs)):
                option_context = choice_json[choices_image_names[i]]
                input_list.append(add_context_input("Option " + str(i+1) + ": " + option_context + "\n"))
        else:
            print("skip")
            continue
            
        output = chat.invoke(input_list)
        
        output = output.content    
        
        print(output)
        
        match = re.search(r'\{.*\}', output, flags=re.DOTALL)
        output = match.group()
        
        output_json = json.loads(output)
        output_json["id"] = case_path
        output_json["answer"] = choices_image_names[output_json["answer"]-1]
        output_json["groundtruth"] = answer_image_name
        
        print(output_json["groundtruth"], output_json["answer"])
        
        output_list.append(output_json)
        
        sample_output = json.dumps(output_json, indent=4)
                
        with open(save_path, "w") as out_file:
            out_file.write(sample_output)
        
        sample_save_path_list.append(save_path)
    
    main_output = json.dumps(output_list)
    
    with open(main_save_path, "w") as out_file:
        out_file.write(main_output)
    
    for save_path in sample_save_path_list:
        os.remove(save_path)
