import json
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Calculate accuracy scores from MMIU evaluation results')
parser.add_argument('--directories', type=str, nargs='+', default=['./results'], help='Directories containing results (default: ./results)')
parser.add_argument('--models', type=str, nargs='+', default=None, help='Model names to evaluate (default: all models found)')
parser.add_argument('--output', type=str, default='./Accuracy_data_all.csv', help='Output CSV file path')
args = parser.parse_args()

directorys = args.directories

# Initialize global DataFrames to store data
global_accuracy_df = pd.DataFrame()

for directory in directorys:
    if not os.path.exists(directory):
        print(f"Warning: Directory '{directory}' does not exist, skipping...")
        continue
    
    tasknames = sorted(os.listdir(directory))

    # Default model names if not specified
    if args.models:
        modelnames = args.models
    else:
        modelnames = ['GPT4o','Claude3','Gemini','Gemini1.0','Llava-interleave','Mantis','InternVL2','internvl1.5-chat','qwen_chat', 'qwen_base', 'idefics_9b_instruct','flamingov2', 'deepseek_vl_1.3b', 'XComposer2_1.8b', 'deepseek_vl_7b', 'idefics2_8b', 'XComposer2', 'qwen3-vl']
    # modelnames = ['Llava-interleave']
    # Initialize dictionaries to store data
    accuracy_data = {modelname: [] for modelname in modelnames}

    for taskname in tasknames:
        path = os.path.join(directory, taskname)
        for modelname in modelnames:
            json_path = os.path.join(path, modelname, 'metadata_info_choice.json')

            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
            else:
                print('no json: ', taskname,modelname)
                accuracy_data[modelname].append(None)
                continue

            

            cnt = 0
            correct = 0
            cnt_z = 0

            for i in range(len(data)):
                data_tmp = data[i]
                flag = True
                if data_tmp[f'{modelname}_choice'].strip() == 'GPT error':
                    print(modelname, taskname, 'GPT error')
                    continue
                
                if data_tmp["output"] == None:
                    flag = False
                    continue
                gt = data_tmp["output"].strip().lower()

                if flag == False:
                    continue

                cnt += 1

                if data_tmp[f'{modelname}_choice'].strip().lower() in gt:
                    correct += 1


            accuracy_data[modelname].append(correct / cnt)
            print(correct / cnt, taskname, modelname)

                

    # Convert dictionaries to DataFrames
    accuracy_df = pd.DataFrame(accuracy_data, index=tasknames)

    # Append to global DataFrames
    global_accuracy_df = pd.concat([global_accuracy_df, accuracy_df])

# If multiple directories were provided, average results for duplicate task names
if len(directorys) > 1:
    # Group by task name and average across directories (skip NaN values)
    global_accuracy_df = global_accuracy_df.groupby(global_accuracy_df.index).mean(skipna=True)

# Calculate the overall average for each model (skip NaN values)
global_accuracy_df.loc['Overall'] = global_accuracy_df.mean(skipna=True)

# Print accuracy results
print("\n" + "="*80)
print("ACCURACY RESULTS")
print("="*80)
print(global_accuracy_df.to_string())
print("="*80)

# Save global DataFrames to CSV files
global_accuracy_df.to_csv(args.output)

print(f"\nAccuracy results have been saved to: {args.output}")


