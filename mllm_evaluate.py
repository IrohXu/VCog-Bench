import argparse
import os
import json

parser = argparse.ArgumentParser()

parser.add_argument('--json-path', type=str, default='./results/gpt4o-0shot-marsvqa.json')

if __name__ == "__main__":
    args = parser.parse_args()
    
    json_path = args.json_path
    with open(json_path, "r") as f:
        json_data = json.load(f)
        
        correct = 0
        total_num = 0
        for case in json_data:
            if case["answer"] == case["groundtruth"]:
                correct += 1
            total_num += 1
        
        print(float(correct / total_num))






