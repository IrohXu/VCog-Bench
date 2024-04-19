from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI
import base64

chat = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=256)

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
    img_path = f"/home/xucao2/VLM_experiment/VCog/data/question/7/q.jpg"
    encoded_question_img = encode_image(img_path)
    img_path = f"/home/xucao2/VLM_experiment/VCog/data/question/7/1.jpg"
    encoded_a_img = encode_image(img_path)
    img_path = f"/home/xucao2/VLM_experiment/VCog/data/question/7/2.jpg"
    encoded_b_img = encode_image(img_path)
    img_path = f"/home/xucao2/VLM_experiment/VCog/data/question/7/3.jpg"
    encoded_c_img = encode_image(img_path)
    img_path = f"/home/xucao2/VLM_experiment/VCog/data/question/5/4.jpg"
    encoded_d_img = encode_image(img_path)
    
    system_template = load_prompt(f"./vcog_prompt.txt")
    response_format = load_prompt(f"./vcog_response_format.txt")
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    
    system_message = system_message_prompt.format(
        response_format=response_format,
    )
    
    print(system_message)

    output = chat.invoke(
        [
            system_message,
            add_image_input(encoded_question_img, "Question:"),
            add_image_input(encoded_a_img, "Choice A:"),
            add_image_input(encoded_b_img, "Choice B:"),
            add_image_input(encoded_c_img, "Choice C:"),
            add_image_input(encoded_d_img, "Choice D:"),
        ]
    )
    
    print(output)
    
    print(output.content)




