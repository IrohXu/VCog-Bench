from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI
import base64
import os
import json

chat = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=512)

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
                    "url": f"data:image/jpeg;base64,{img}",
                    "detail": "auto",
                },
            },
        ]
    )
    return out

if __name__ == "__main__":
    example_path = "/home/xucao2/VLM_experiment/VCog/dataset/task1/tf1/pd"
    
    for sample in os.listdir(example_path):
        sample_path = os.path.join(example_path, sample)
        choice_img_path = os.path.join(sample_path, "choice", "image")
        choice_txt_path = os.path.join(sample_path, "choice", "text")
        if not os.path.exists(choice_txt_path):
            os.mkdir(choice_txt_path)
        choices_image_names = os.listdir(choice_img_path)
        encoded_choice_imgs = []
        choice_imgs = []
        for img_path in choices_image_names:
            choice_imgs.append(img_path)
            encoded_choice_imgs.append(encode_image(os.path.join(choice_img_path, img_path)))
        
        if os.path.exists(os.path.join(choice_txt_path, "annotation.json")):
            print(os.path.join(choice_txt_path, "annotation.json"))
            continue
        
        system_template = load_prompt(f"/home/xucao2/VLM_experiment/VCog/scripts/vi_prompt.txt")
        response_format = load_prompt(f"/home/xucao2/VLM_experiment/VCog/scripts/vi_response_format.txt")
        
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        
        system_message = system_message_prompt.format(
            response_format=response_format,
        )
        
        input_list = [system_message]
            
        for i in range(0, len(encoded_choice_imgs)):
            choice_img = encoded_choice_imgs[i]
            input_list.append(add_image_input(choice_img, "Title: " + choice_imgs[i]))
        
        output = chat.invoke(input_list)
        
        print(output.content)
        
        output_json = json.loads(output.content)
        output_json = json.dumps(output_json, indent=4)
        
        with open(os.path.join(choice_txt_path, "annotation.json"), "w") as out_file:
            out_file.write(output.content)
        
    