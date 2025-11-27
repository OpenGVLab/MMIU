import os
import json
import base64
import argparse
import random
from multiprocessing import Pool
from tqdm import tqdm
from openai import OpenAI
from PIL import Image
import io

# Configure API client
base_url = os.getenv('OPENAI_API_BASE', 'https://api.fireworks.ai/inference/v1')
api_key = os.getenv('FIREWORKS_API_KEY', None)

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)

def encode_image(image_path):
    """Encode image to base64."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Save to bytes buffer
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=95)
            buffer.seek(0)
            
            return base64.b64encode(buffer.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def call_qwen3_vl(image_paths, question, model_name="qwen3-vl", interleave_text=False):
    """Call qwen3-vl via OpenAI-compatible API."""
    content = []
    
    # Random text snippets to interleave between images
    interleave_texts = [
        "Here is image",
        "This image shows",
        "Looking at this image",
        "In this image we can see",
        "This picture contains",
        "Image",
    ]
    
    # Add images (with optional interleaved text)
    for i, image_path in enumerate(image_paths):
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return 'image error'
        
        base64_image = encode_image(image_path)
        if base64_image is None:
            return 'image error'
        
        # Add interleaved text before each image (except the first)
        if interleave_text and i > 0:
            interleave_text_snippet = random.choice(interleave_texts) + f" {i+1}."
            content.append({
                "type": "text",
                "text": interleave_text_snippet
            })
        
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })
    
    # Add text question
    content.append({
        "type": "text",
        "text": question
    })
    
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
        )
        answer = response.choices[0].message.content
        print(f"ASSISTANT: {answer}")
        return answer
    except Exception as e:
        print(f"Model error: {e}")
        return 'model error'

parser = argparse.ArgumentParser(description='Run Qwen3-VL inference on MMIU dataset')
parser.add_argument('--json_path', type=str, default='all.json', help='Path to all.json file')
parser.add_argument('--limit', type=int, default=None, help='Limit to first N rows')
parser.add_argument('--sample', type=int, default=None, help='Random sample of N rows')
parser.add_argument('--tasks', type=str, nargs='+', default=None, help='Filter by specific task names')
parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers (default: 1)')
parser.add_argument('--output-dir', type=str, default='../results', help='Output directory for results (default: ../results)')
parser.add_argument('--interleave-random-text', action='store_true', help='Interleave random text snippets between images (just for testing)')
args = parser.parse_args()

json_path = args.json_path

tasks_exist = ['person_reid', 'multiple_image_captioning', 'spot_the_similarity', 'face_retrieval', 'sketch2image_retrieval', 'handwritten_retrieval', 'spot_the_diff', 'image2image_retrieval', 'vehicle_retrieval', 'text2image_retrieval',
'general_action_recognition', 'video_captioning', 'next_img_prediction', 'temporal_ordering', 'meme_vedio_understanding', 'action_quality_assessment', 'temporal_localization', 'mevis',
'ravens_progressive_matrices', 'threed_indoor_recognition', 'point_tracking', 'threed_cad_recognition', 'single_object_tracking']

# Model name - can be overridden via environment variable
model_name = os.getenv('MODEL', 'qwen3-vl')

if not os.path.exists(json_path):
    print(f"Error: {json_path} not found!")
    print("Please create all.json with the MMIU dataset.")
    exit(1)

with open(json_path, 'r') as f:
    data_all = json.load(f)

# Apply filters
original_count = len(data_all)
if args.tasks:
    data_all = [d for d in data_all if d.get('task') in args.tasks]
    print(f"Filtered to {len(data_all)} rows matching tasks: {args.tasks}")

if args.sample:
    random.seed(args.seed)
    data_all = random.sample(data_all, min(args.sample, len(data_all)))
    print(f"Sampled {len(data_all)} rows (seed={args.seed})")
elif args.limit:
    data_all = data_all[:args.limit]
    print(f"Limited to first {len(data_all)} rows")

# Seed random for interleaved text (if enabled)
if args.interleave_random_text:
    random.seed(args.seed)

if original_count != len(data_all):
    print(f"Processing {len(data_all)} rows (out of {original_count} total)")

def process_single_item(args_tuple):
    """Process a single task_data item. Used for parallel processing."""
    task_data, model_name, tasks_exist, interleave_random_text = args_tuple
    
    context = task_data["context"]
    question = task_data["question"]
    
    tmp = []
    image_flag = True
    
    for image_path in task_data["input_image_path"]:
        tmp.append(image_path)
        if not os.path.exists(image_path):
            image_flag = False
            break
    
    if image_flag == False:
        response = 'image none'
        task_data[model_name] = response
        print(f"{model_name}, {task_data.get('task', 'unknown')}, {len(tmp)}: {response}")
        return task_data
    
    try:
        if task_data['task'] in tasks_exist:
            question_formatted = question + '\n' + context
        else:
            question_formatted = context + '\n' + question
        question_formatted = question_formatted + '\nPlease answer the option directly like A,B,C,D...'
        
        response = call_qwen3_vl(tmp, question_formatted, model_name=model_name, interleave_text=interleave_random_text)
        task_data[model_name] = response
        print(f"{model_name}, {task_data.get('task', 'unknown')}, {len(tmp)}: {response}")
    except Exception as e:
        response = 'model error or image error'
        task_data[model_name] = response
        print(f"{model_name}, {task_data.get('task', 'unknown')}, {len(tmp)}: {response}")
        print(f"Exception: {e}")
    
    return task_data

# Process items (sequentially or in parallel)
if args.workers > 1:
    print(f"Processing {len(data_all)} items with {args.workers} parallel workers...")
    if args.interleave_random_text:
        print("Interleaving random text between images enabled")
    # Create argument tuples for each item
    process_args = [(task_data, model_name, tasks_exist, args.interleave_random_text) for task_data in data_all]
    
    # Process in parallel with progress bar
    with Pool(processes=args.workers) as pool:
        processed_data = list(tqdm(
            pool.imap(process_single_item, process_args),
            total=len(process_args),
            desc="Processing"
        ))
else:
    print(f"Processing {len(data_all)} items sequentially...")
    if args.interleave_random_text:
        print("Interleaving random text between images enabled")
    processed_data = [
        process_single_item((task_data, model_name, tasks_exist, args.interleave_random_text))
        for task_data in tqdm(data_all, desc="Processing")
    ]

# Organize results by task
results_by_task = {}
for task_data in processed_data:
    task_name = task_data.get('task', 'unknown')
    if task_name not in results_by_task:
        results_by_task[task_name] = []
    results_by_task[task_name].append(task_data)

# Save results organized by task
base_output_dir = os.path.abspath(args.output_dir)  # Resolve to absolute path to avoid issues
if not os.path.exists(base_output_dir):
    os.makedirs(base_output_dir)

# Extract basename for directory structure (in case model_name is a full path)
model_dir_name = os.path.basename(model_name) if os.path.sep in model_name else model_name

for task_name, task_results in results_by_task.items():
    task_dir = os.path.join(base_output_dir, task_name)
    model_dir = os.path.join(task_dir, model_dir_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    output_path = os.path.join(model_dir, 'metadata_info.json')
    with open(output_path, 'w') as f:
        json.dump(task_results, f)
    print(f"Saved {len(task_results)} results for task '{task_name}' to {output_path}")

print(f"\nAll results saved to: {os.path.abspath(base_output_dir)}")

