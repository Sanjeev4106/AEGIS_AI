
import tensorflow_hub as hub
import numpy as np
import csv
import io

def list_classes():
    # Load YAMNet model
    print("Loading YAMNet model...")
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    
    # Get class map from model
    class_map_path = model.class_map_path().numpy().decode('utf-8')
    print(f"Class map path: {class_map_path}")
    
    with open(class_map_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader) # skip header
        with open("yamnet_classes.txt", "w") as out:
            for i, row in enumerate(reader):
                # row is [index, display_name, mid]
                # YAMNet index matches line number (0-indexed)
                out.write(f"{i}: {row[1]}\n")
    print("Dumped classes to yamnet_classes.txt")

if __name__ == "__main__":
    list_classes()
