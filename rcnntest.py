import os
import re

# Function to extract class names from label map file
def extract_class_names(label_map_path):
    class_names = []
    with open(label_map_path, 'r') as f:
        for line in f:
            match = re.match(r"name: '(\w+)'", line)
            if match:
                class_names.append(match.group(1))
    return class_names

# Function to generate the pbtxt file
def generate_pbtxt(class_names, output_file):
    with open(output_file, 'w') as f:
        for i, class_name in enumerate(class_names, start=1):
            f.write("item {\n")
            f.write(f"\tid: {i}\n")
            f.write(f"\tname: '{class_name}'\n")
            f.write("}\n")

# Extract label map path from config file
config_file = 'C:/Users/dipto/Desktop/retail iter rcnn/faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8/pipeline.config'
with open(config_file, 'r') as f:
    config_content = f.read()
    match = re.search(r'label_map_path: "(.*)"', config_content)
    if match:
        label_map_path = match.group(1)
    else:
        raise ValueError("Label map path not found in the config file.")

# Extract class names from label map file
class_names = extract_class_names(label_map_path)

# Generate pbtxt file
output_pbtxt_file = 'label_map.pbtxt'
generate_pbtxt(class_names, output_pbtxt_file)

print("label_map.pbtxt generated successfully.")
