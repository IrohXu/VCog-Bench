import argparse
import os
import json

parser = argparse.ArgumentParser()

parser.add_argument('--llm', type=str, default='claude3')
parser.add_argument('--main-path', type=str, default='/home/xucao2/VLM_experiment/VCog/dataset/task1/tf2/pd')

if __name__ == "__main__":
    args = parser.parse_args()
    
    llm_name = args.llm
    main_path = args.main_path
    
    correct = 0
    total_num = 0
    for case_path in os.listdir(main_path):
        result_path = os.path.join(main_path, case_path, "result")

        json_path = os.path.join(result_path, llm_name + ".json")
        
        f = open(json_path)
        json_data = json.load(f)
        
        if json_data["Answer"] == json_data["Groundtruth"]:
            correct += 1
        total_num += 1
    
    print(float(correct / total_num))






