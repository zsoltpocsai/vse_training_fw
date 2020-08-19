"""Splits dataset in two by given rules.

In the given destination the proper folder structure is going to be created
automatically. The `labels.csv` files are going to be updated on the source
and created on the destination side.

The rules are the following:
- The number of images to move.
- The minimum positive pairs the new dataset must contain.
    It means that if you split the dataset with `min_positive_pairs = 100`
    then it's guaranteed that there will be at least 100 pair of sample
    images in the new dataset which belong to the same whale.

Args:
    dir_to_split: Path of the source dataset directory.
    dest_dir: Path of the directory where the splitted portion of the
        dataset is going to be moved.
"""
import sys, os, csv, argparse, random
sys.path.append(os.getcwd())
from utils import data_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_to_split")
    parser.add_argument("dest_dir")

    args = parser.parse_args()

    img_dir = args.dir_to_split + "/images"
    dest_img_dir = args.dest_dir + "/images"
    labels_file = args.dir_to_split + "/labels.csv"
    dest_labels_file = args.dest_dir + "/labels.csv"

    if not os.path.exists(dest_img_dir):
        os.makedirs(dest_img_dir)

    # Set the rules here
    # --------------------------------

    image_to_move = 200
    min_positive_pairs = 50

    min_whale_image = 2

    #---------------------------------
    
    label_map = data_map.csv2lblmap(labels_file)
    image_map = data_map.csv2imgmap(labels_file)

    rnd = random.Random()
    rnd.seed()
    
    # whales in the positive pool must have at least 3 images
    if min_whale_image <= 3:
        positive_pool_treshold = 3
    else:
        positive_pool_treshold = min_whale_image

    positive_pool = [
        label for label in label_map.keys() \
        if len(label_map[label]) >= positive_pool_treshold
    ]
    rnd.shuffle(positive_pool)

    # getting the positive pairs
    while min_positive_pairs > 0:
        label = rnd.choice(positive_pool)

        for _ in range(2):
            image = rnd.choice(label_map[label])

            image_path = os.path.join(img_dir, image)
            dest_img_path = os.path.join(dest_img_dir, image)
            os.rename(image_path, dest_img_path)
            
            label_map[label].remove(image)

        if len(label_map[label]) < 3:
            positive_pool.remove(label)

        min_positive_pairs -= 1
        image_to_move -= 2

    image_list = [img for img in os.listdir(img_dir)]
    rnd.shuffle(image_list)

    # select random for the rest
    while image_to_move > 0:
        image = image_list.pop()

        label = image_map[image]
        if len(label_map[label]) < min_whale_image:
            continue

        image_path = os.path.join(img_dir, image)
        dest_img_path = os.path.join(dest_img_dir, image)
        os.rename(image_path, dest_img_path)

        label_map[label].remove(image)
        image_to_move -= 1

    # create and modify the labels.csv files
    data_map.create_csv(labels_file, img_dir, image_map)
    data_map.create_csv(dest_labels_file, dest_img_dir, image_map)
