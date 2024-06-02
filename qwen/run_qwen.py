import argparse
import json
import os
import re

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset-path", type=str, default="/media/mayson/Data/datasets/vcog/VCog/benchmark/marsvqa"
)
parser.add_argument("--output_path", type=str, default="/media/mayson/Data/github/VCog/qwen/results")
parser.add_argument("--system-template", type=str, default="/media/mayson/Data/github/VCog/qwen/prompt_raven.txt")

args = parser.parse_args()


def load_prompt(prompt_path):
    with open(prompt_path, "r") as f:
        prompt = f.read()
    return prompt


if __name__ == "__main__":
    torch.manual_seed(1234)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen-VL-Chat", trust_remote_code=True, device_map="cuda"
    ).eval()
    # Specify the generation config
    model.generate_config = GenerationConfig.from_pretrained(
        "Qwen/Qwen-VL-Chat", trust_remote_code=True
    )

    main_path = args.dataset_path
    output_list = []
    main_save_path = os.path.join(args.output_path, "qwen_main.json")
    # load existing results
    if os.path.exists(main_save_path):
        with open(main_save_path, "r") as f:
            output_list = json.load(f)

    for case_path in tqdm(os.listdir(main_path)):
        # print(f"Processing {case_path}")
        case_path = os.path.join(main_path, case_path)
        question_img_path = os.path.join(case_path, "question", "image")
        answer_img_path = os.path.join(case_path, "answer", "image")
        choice_img_path = os.path.join(case_path, "choice", "image")
        choice_text_path = os.path.join(case_path, "choice", "text", "annotation.json")
        choice_json = json.load(open(choice_text_path))
        result_path = os.path.join(args.output_path, case_path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        save_path = os.path.join(result_path, "qwen.json")

        answer_image_name = os.listdir(answer_img_path)[0]
        choices_image_names = os.listdir(choice_img_path)
        question_path_name = os.listdir(question_img_path)[0]
        question_img_path = os.path.join(question_img_path, question_path_name)

        system_prompt = load_prompt(args.system_template)
        input_list = []

        for i in range(len(choices_image_names)):
            option_context = choice_json[choices_image_names[i]]
            input_list.append("Option " + str(i + 1) + ": " + option_context + "\n")
        text = system_prompt + "\n" + " ".join(input_list)
        response, _ = model.chat(tokenizer, query=text, history=None)

        try:
            mt = re.search(r"\d+", response)
            n = mt.group()
        except Exception as e:
            print(f"Error: {e}")
            print(f"Response: {response}")
            print(f"Text: {text}")
            n = 1

        # print(f"Response: {n}")
        query = tokenizer.from_list_format([{"image": question_img_path}, {"text": text}])

        output_json = {}
        output_json["case_path"] = case_path
        output_json["answer"] = choices_image_names[int(n) - 1]
        output_json["groundtruth"] = answer_image_name
        # print(f"Groundtruth: {choices_image_names.index(answer_image_name) + 1}")
        # print("--------------------")
        output_list.append(output_json)
        sample_output = json.dumps(output_json, indent=2)
        with open(save_path, "w") as out_file:
            out_file.write(sample_output)

        main_output = json.dumps(output_list)
        with open(main_save_path, "w") as out_file:
            out_file.write(main_output)

    # Compute accuracy
    correct = 0
    total = len(output_list)
    for item in output_list:
        if item["answer"] == item["groundtruth"]:
            correct += 1
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Correct: {correct}")
    print(f"Total: {total}")
