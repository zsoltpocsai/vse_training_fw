"""Test refers to testing the modell in the identification task.

Can be run from the command line or used as a module by calling the 
`run` method.
"""

import argparse, os, math, csv
import tensorflow as tf
import numpy as np

from utils import knn, data_generator, data_map, config


def run(model_path,
        base_vectors_dir,
        dataset_dir,
        input_shape=(220, 220, 1),
        verbose=1):
    """Runs the testing process.

    It produces the embedded vectors of the dataset given at the 
    location of `dataset_dir` and tries to identify them based on
    the vectors located at `base_vectors_dir`. The identification
    works with k-NN search algorithm. The k value can be changed
    in the code. The result will be saved into the 
    `base_vectors_dir` directory.

    Args:
        model_path: Full path of the CNN model.
        base_vectors_dir: Path of the directory where the embedded vectors
            populating the vector-space can be found. It must contain a
            `vecs.tsv` and a `metas.tsv` file.
        dataset_dir: Path of the dataset root directory which contains the 
            images needed to be indentified.
        input_shape: Tuple. The shape of the model's input layer.
        verbose: If set to 1, prints some info to the command line.
            At 0, it will keep quiet. Default is 1.
    """

    # Loading the model

    model = tf.keras.models.load_model(model_path, compile=False)

    if verbose == 1:
        print("Model has been successfully loaded.")

    # Prepare the data generator
    # and calculate the steps

    labels_file = os.path.join(dataset_dir, "labels.csv")

    data_with_labels = data_map.read_csv(labels_file, image_full_path=True)
    data = data_generator.enumerate_labels(data_with_labels)

    batch_size = 30
    generator = data_generator.for_test(data, input_shape, batch_size)

    data_size = len(data[0]) # data = (x_data, y_data)
    steps = math.ceil(data_size / batch_size)
    
    # Make a kNN search

    if verbose == 1:
        print(f"Identifying {data_size} samples...")

    embeddings = model.predict_generator(
        generator,
        steps,
        verbose=1
    )

    print("Calculating nearest neighbors...")

    # Change the k value here
    k = 4

    predictions = knn.get_predictions(base_vectors_dir, embeddings, k=k)

    # Save the result
    
    good_guesses = 0

    images = list(np.squeeze(data_with_labels[0]))
    image_map = data_map.get_image_map(data_with_labels)

    with open(os.path.join(base_vectors_dir, "submission.csv"), "w", newline="") as file:

        writer = csv.writer(file)
        header = ["Image", "Id"]
        writer.writerow(header)

        for i in range(len(predictions)):

            try:
                image_name = os.path.basename(images[i])
            except:
                pass

            label = image_map[images[i]]

            if label in predictions[i]:
                good_guesses += 1

            # only for kaggle submission
            # row = [image_name, " ".join(["new_whale"] + predictions[i])]

            row = [image_name + f"({label})", " ".join(predictions[i])]
            writer.writerow(row)

    result = math.floor((good_guesses / data_size) * 100)

    with open(os.path.join(base_vectors_dir, "eval.txt"), "w") as file:

        file.write(f"Evaluated dataset: {dataset_dir}\n")
        file.write(f"Evaluation: {result}%\n")

    print(f"\n** Result: {result}% **\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("run_config_file")
    args = parser.parse_args()

    kwargs = config.get_test_args(args.run_config_file)

    run(**kwargs)
