"""Looks for images which are exact copies of each other.

Requires:
    ImageHash ^4.0.0
    Pillow ^7.0.0
"""

import os, csv, argparse
import imagehash
from PIL import Image


def get_duplications(images_dir):
    """Identifies image duplications.
    Returns:
        A list of tuples, where each tuple contains the imagenames recognized
        as duplicates of each other.
    """
    images_list = os.listdir(images_dir)
    images_full_path_list = list(
        map(lambda image_name: os.path.join(images_dir, image_name), images_list)
    )
    
    phash_dict = {}

    for image_path in images_full_path_list:
        with Image.open(image_path) as image:
            phash = imagehash.phash(image)
        
        if phash not in phash_dict:
            phash_dict.update({phash: [image_path]})
        else:
            phash_dict[phash].append(image_path)

    duplications = []

    for phash in list(phash_dict.keys()):
        if len(phash_dict[phash]) > 1:
            duplications.append(
                tuple(map(lambda x: os.path.basename(x), phash_dict[phash]))
            )

    return duplications


def save(duplications_list, to_file):
    """Saves the findings into a file.
    """
    duplications = duplications_list

    with open(to_file, "w") as file:
        file.write(f"Total: {len(duplications)}\n")
        for item in duplications:
            for image in item:
                file.write(f" {image} ")
            file.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images_dir")
    parser.add_argument("save_path")

    args = parser.parse_args()

    save(get_duplications(args.images_dir), args.save_path)
