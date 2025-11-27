import json
import os
import base64
import time
import argparse
from openai import OpenAI
from multiprocessing import Pool
import re

def remove_punctuation(text):
    return re.sub(r'^[.,()]+|[.,()]+$', '', text)

# Default API configuration
default_base_url = 'https://api.fireworks.ai/inference/v1'
default_api_key = os.getenv('FIREWORKS_API_KEY', None)

def build_prompt(question, options, prediction):
    tmpl = (
        "You are an AI assistant who will help me to match an answer with several options of a single-choice question. "
        "You are provided with a question, several options, and an answer, and you need to find which option is most similar to the answer. "
        "If the meaning of all options are significantly different from the answer, output Z. "
        "When the options are mostly numbers, if the model outputs numbers in the same format, please do not be too precise and try to match an answer as much as possible. "\
        "Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. \n"
        "Example 1: \n"
        "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\nAnswer: a cute teddy bear\nYour output: A\n"
        "Example 2: \n"
        "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\nAnswer: Spider\nYour output: Z\n"
        "Example 3: \n"
        "Question: {}?\nOptions: {}\nAnswer: {}\nYour output: "
    )
    return tmpl.format(question, options, prediction)


def process_data(args_tuple):
    data_tmp, modelname, base_url, api_key = args_tuple
    client = OpenAI(
    # base_url='https://kkkc.net/v1',
    # api_key='sk-YJaHfazVSf2WDkAl1bAdE17bF3Ae4923Ba888293B31d13C4',
    base_url=base_url,
    api_key=api_key,
    )

    options = data_tmp['options']
    question = data_tmp['question']
    
    # Try to find the prediction key - the JSON might have full path or basename as key
    prediction_key = None
    if modelname in data_tmp:
        prediction_key = modelname
    else:
        # Try basename if modelname is a full path
        basename = os.path.basename(modelname) if os.path.sep in modelname else modelname
        if basename in data_tmp:
            prediction_key = basename
        else:
            # Try to find any key that ends with the basename (for full paths in JSON)
            for key in data_tmp.keys():
                if key.endswith(basename) or os.path.basename(key) == basename:
                    prediction_key = key
                    break
    
    if prediction_key is None or prediction_key not in data_tmp:
        available_keys = [k for k in data_tmp.keys() if k not in ['options', 'question', 'context', 'input_image_path', 'task', 'output', 'visual_input_component', 'source']]
        print(f"Warning: Key for model '{modelname}' not found in data. Available model keys: {available_keys}")
        data_tmp[f'{modelname}_choice'] = 'Z'
        return data_tmp
    
    prediction = data_tmp[prediction_key].strip()

    if modelname == 'Claude3' and "copyrighted material" in prediction:
        data_tmp[f'{modelname}_choice'] = 'Z'
        return data_tmp
    if prediction == 'image none' or prediction == 'model error or image error' or prediction == 'image error' or prediction == 'model error' or prediction == "":
        data_tmp[f'{modelname}_choice'] = 'Z'
        return data_tmp
    if '\u00a0' in prediction:
        prediction = prediction.replace('\u00a0','')


    prediction = remove_punctuation(prediction.strip())
    
    if prediction.strip().lower() not in ['a','b','c','d','e','f','g','h','i','j','k','l','m','n']:



        content = build_prompt(question,options,prediction)

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": content},
                        ],
                    }
                ],
                max_tokens=512,
            )
            # print(response.choices[0].message.content)
            grading = response.choices[0].message.content
            
        except Exception as e:
            print('errror: ', e)
            # grading = str(e)
            grading = 'GPT error'


        data_tmp[f'{modelname}_choice'] = grading.strip()
        print(modelname,': ',data_tmp[f'{modelname}_choice'])
        return data_tmp
    else:
        data_tmp[f'{modelname}_choice'] = prediction.strip()
        print(modelname,': ',data_tmp[f'{modelname}_choice'])
        return data_tmp



def main():
    parser = argparse.ArgumentParser(description='Convert model predictions to multiple choice answers')
    parser.add_argument('--directories', type=str, nargs='+', default=['./results'], help='Directories containing results (default: ./results)')
    parser.add_argument('--models', type=str, nargs='+', default=['qwen3-vl'], help='Model names to evaluate (default: qwen3-vl)')
    parser.add_argument('--workers', type=int, default=10, help='Number of parallel workers (default: 10)')
    parser.add_argument('--base-url', type=str, default=None, help=f'OpenAI API base URL (default: {default_base_url})')
    parser.add_argument('--api-key', type=str, default=None, help='OpenAI API key (default: from FIREWORKS_API_KEY env var)')
    args = parser.parse_args()
    
    modelnames = args.models
    directorys = args.directories
    base_url = args.base_url or default_base_url
    api_key = args.api_key or default_api_key
    
    if not api_key:
        print("Error: API key not provided. Set FIREWORKS_API_KEY environment variable or use --api-key")
        return
   
    for directory in directorys:
        if not os.path.exists(directory):
            print(f"Warning: Directory '{directory}' does not exist, skipping...")
            continue
            
        tasknames = os.listdir(directory)
        for taskname in tasknames:
            for modelname in modelnames:
                path = os.path.join(directory, taskname, modelname)

                print(taskname, modelname)
                json_path = os.path.join(path, 'metadata_info.json')

                if not os.path.exists(json_path):
                    print(json_path, ' not exist')
                    continue

                output_json_path = os.path.join(path, 'metadata_info_choice.json')
                if os.path.exists(output_json_path):
                    print(output_json_path, ' already have')
                    continue

                with open(json_path, 'r') as f:
                    data = json.load(f)

                # Pack data with modelname, base_url, and api_key
                data_with_args = [(data_tmp, modelname, base_url, api_key) for data_tmp in data]

                pool = Pool(processes=args.workers)
                result = pool.map(process_data, data_with_args)
                pool.close()
                pool.join()

                with open(output_json_path, 'w') as f:
                    json.dump(result, f)

                print(taskname, modelname, 'OK')


if __name__ == '__main__':
    main()

