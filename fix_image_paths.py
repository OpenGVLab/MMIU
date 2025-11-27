import json
import os

# Load the JSON file
with open('VLMEvalKit/all.json', 'r') as f:
    data = json.load(f)

# Update image paths
updated = 0
for entry in data:
    old_paths = entry.get('input_image_path', [])
    if isinstance(old_paths, list):
        # Remove the category prefix (e.g., "Low-level-semantic/") and update path
        new_paths = []
        for p in old_paths:
            # Path format: ./Low-level-semantic/task_name/image.jpg
            # We need: ../MMIU-Benchmark/task_name/image.jpg
            parts = p.split('/')
            if len(parts) >= 3:
                # Skip the category directory (parts[1]) and keep task/image
                new_path = '../MMIU-Benchmark/' + '/'.join(parts[2:])
            else:
                # Fallback: just replace ./ with ../MMIU-Benchmark/
                new_path = p.replace('./', '../MMIU-Benchmark/')
            new_paths.append(new_path)
        entry['input_image_path'] = new_paths
        updated += 1

# Save updated JSON
with open('VLMEvalKit/all.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f'Updated {updated} entries with corrected image paths')

# Verify one path
if len(data) > 0:
    sample_path = data[0]['input_image_path'][0]
    print(f'\nSample path: {sample_path}')
    print(f'Path exists (from VLMEvalKit/): {os.path.exists(os.path.join("VLMEvalKit", sample_path))}')

