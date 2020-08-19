"""Gives some basic analisys of the dataset.

Counts the number of images and the number of distinct whales in the
dataset. Also provides a histogram which tells the amount of
whales having an x number of image. So you can check how many whale
exist in the dataset which has only 1 image, 2 images, etc.

Saves the information into the directory of the dataset in two files:
    info.txt - The number of images and whales.
    histogram.csv - As described above. Has two columns separated by ','.
        First column is an occurrance of the number of images a whale has.
        Second column is the number of whales having the amount of image
        in the first column.

Args:
    dataset_dir: The path of dataset directory. The dataset must have
        the directory structure described in the usage documentation.
"""

import os, argparse, csv


def _create_mapping(labels_file):
    img2whale = {}

    csv_file = open(labels_file, "r")
    reader = csv.reader(csv_file)
    # skip first row
    next(reader)

    for row in reader:
        img_name = row[0]
        whale = row[1]
        img2whale.update({img_name: whale})

    csv_file.close()
    return img2whale


def _img_per_whale(img2whale):
    img_per_whale = {}

    for img in list(img2whale.keys()):
        whale = img2whale[img]
        if whale in img_per_whale:
            img_per_whale[whale] += 1
        else:
            img_per_whale.update({whale: 1})

    return img_per_whale


def _create_histogram(img_per_whale):
    histogram = {}

    for whale in list(img_per_whale.keys()):
        amount = img_per_whale[whale]
        if amount in histogram:
            histogram[amount] += 1
        else:
            histogram.update({amount: 1})

    return histogram


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir")
    args = parser.parse_args()

    labels_file = os.path.join(args.dataset_dir, "labels.csv")

    img2whale = _create_mapping(labels_file)
    images = list(img2whale.keys())
    whales = set(img2whale.values())

    print("Number of images:", len(images))
    print("Number of classes:", len(whales))

    with open(os.path.join(args.dataset_dir, "info.txt"), "w") as file:
        file.write(f"Number of images: {len(images)}\n")
        file.write(f"Number of whales: {len(whales)}\n")

    histogram = _create_histogram(_img_per_whale(img2whale))

    with open(os.path.join(args.dataset_dir, "histogram.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Number of images", "Number of whales"])
        for num_of_images in sorted(list(histogram.keys())):
            num_of_whales = histogram[num_of_images]
            writer.writerow([num_of_images, num_of_whales])
    