"""Filters the dataset by given rules.

After removing whales and images from the directory, it recreates the 
`labels.csv` file so it mirrors the changes.

Args:
    dataset_dir: Path of the dataset directory. Must contain an `images`
        subdirectory and a `labels.csv` file.
    min_img_num: The number of minimum images a whale must have. If a whale 
        found where this requirement fails then the whale id and all its 
        images will be deleted from the dataset. 
"""
import sys, os, csv, argparse
sys.path.append(os.getcwd())
from utils import data_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir")
    parser.add_argument("min_img_num")

    args = parser.parse_args()

    img_dir = os.path.join(args.dataset_dir, "images")
    labels_file = os.path.join(args.dataset_dir, "labels.csv")

    # Filter rules
    MIN_IMAGE_NUMBER = int(args.min_img_num)
    INCLUDE_NEW_WHALE = False

    image_map = data_map.csv2imgmap(labels_file)
    label_map = data_map.csv2lblmap(labels_file)
    
    for whale_id in list(label_map.keys()):
        whale_images = label_map[whale_id]

        if len(whale_images) < MIN_IMAGE_NUMBER or \
           (whale_id == "new_whale" and not INCLUDE_NEW_WHALE): 
            
            for img in whale_images:
                full_path = os.path.join(img_dir, img)
                os.remove(full_path)

    data_map.create_csv(labels_file, img_dir, image_map)
