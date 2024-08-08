import json
import os
import base64
import time
from openai import OpenAI
from multiprocessing import Pool
import re

def remove_punctuation(text):
    return re.sub(r'^[.,()]+|[.,()]+$', '', text)

client = OpenAI(
    base_url='xx',
    api_key='xx',
)

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


def process_data(args):
    data_tmp, modelname = args
    client = OpenAI(
    # base_url='https://kkkc.net/v1',
    # api_key='sk-YJaHfazVSf2WDkAl1bAdE17bF3Ae4923Ba888293B31d13C4',
    base_url='xx',
    api_key='xx',
    )

    options = data_tmp['options']
    question = data_tmp['question']
    prediction = data_tmp[modelname].strip()

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
    # modelnames = ['internvl1.5-chat']
    # modelnames = ['Gemini','Gemini1.0']
    # modelnames = ['GPT4o','Gemini','Gemini1.0']
    # modelnames = ['Llava-interleave']
    modelnames = ['Llava-interleave', 'qwen_chat', 'XComposer2', 'deepseek_vl_7b', 'qwen_base', 'XComposer2_1.8b', 'flamingov2', 'deepseek_vl_1.3b', 'internvl1.5-chat', 'idefics2_8b', 'Mantis', 'idefics_9b_instruct']
    directorys = ['xx','xx']
   
    for directory in directorys:
        tasknames = os.listdir(directory)
        for taskname in tasknames:
            
            path = os.path.join(directory,taskname)
            for modelname in modelnames:
                path = os.path.join(directory,taskname)
                path = os.path.join(path,modelname)

                print(taskname,modelname)
                json_path = os.path.join(path,'metadata_info.json')
                


                if not os.path.exists(json_path):
                    print(json_path,' not exist')
                    continue

                # output_json_path = os.path.join(path,'metadata_info_choice.json')
                output_json_path = os.path.join(path,'metadata_info_choice.json')
                # if os.path.exists(output_json_path) or os.path.exists(output_json_path1):
                if os.path.exists(output_json_path):
                    print(output_json_path, ' already have')
                    continue

                with open(json_path,'r') as f:
                    data = json.load(f)

                        # 将data和modelname打包成元组列表
                data_with_modelname = [(data_tmp, modelname) for data_tmp in data]

                

                pool = Pool(processes=10)  # Adjust the number of processes as per your machine's capability
                # result = pool.map(process_data, data, modelname)
                # 使用map方法传递打包后的元组列表
                result = pool.map(process_data, data_with_modelname)
        
                # output_json_path = os.path.join(path,'metadata_info_choice.json')

                with open(output_json_path, 'w') as f:
                    json.dump(result, f)

                print(taskname,modelname,'OK')
            


if __name__ == '__main__':
    main()

