import os
import tarfile
import requests
import pandas as pd
from path import Path
from parameters import *


def downloadData(data_path="/input/speech_commands/"):
    """
    Downloads Google Speech Commands dataset (version0.01)
    :param data_path: Path to download dataset
    :return: None
    """

    dataset_path = Path(os.path.abspath(__file__)).parent.parent + data_path

    datasets = ["train", "test"]
    urls = [
        "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz",
        "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_test_set_v0.01.tar.gz",
    ]

    for dataset, url in zip(datasets, urls):
        dataset_directory = dataset_path + dataset

        # Check if we need to extract the dataset
        if not os.path.isdir(dataset_directory):
            os.makedirs(dataset_path, exist_ok=True)
            file_name = dataset_path + dataset + ".tar.gz"

            # Check if the dataset has been downloaded, else download it
            if os.path.isfile(file_name):
                print("{} already downloaded. Skipping download.".format(file_name))
            else:
                print("Downloading '{}' into '{}' file".format(url, file_name))

                with requests.get(url, stream=True, timeout=60) as data_request:
                    data_request.raise_for_status()
                    with open(file_name + ".part", "wb") as file:
                        for chunk in data_request.iter_content(chunk_size=1 << 20):
                            file.write(chunk)
                os.replace(file_name + ".part", file_name)

            # Extract downloaded file
            print("Extracting {} into {}".format(file_name, dataset_directory))

            if file_name.endswith("tar.gz"):
                with tarfile.open(file_name, "r:gz") as tar:
                    tar.extractall(path=dataset_directory + ".part", filter="data")
                os.replace(dataset_directory + ".part", dataset_directory)
            else:
                print("Unknown format.")
        else:
            print(f"{dataset} data setup complete.")

    print("Input data setup successful.")


def getDataDict(data_path="/input/speech_commands/"):
    """
    Creates a dictionary with train, test, validate and test file names and labels.
    :param data_path: Path to the downloaded dataset
    :return: Dictionary
    """

    data_path = Path(os.path.abspath(__file__)).parent.parent + data_path

    # Get the validation files
    validation_files = open(data_path + "train/validation_list.txt").read().splitlines()
    validation_files = [os.path.normpath(data_path + "train/" + file_name) for file_name in validation_files]

    # Get the dev files
    dev_files = open(data_path + "train/testing_list.txt").read().splitlines()
    dev_files = [os.path.normpath(data_path + "train/" + file_name) for file_name in dev_files]

    # Find train_files as allFiles - {validation_files, dev_files}
    all_files = []
    for root, dirs, files in os.walk(os.path.normpath(data_path + "train/")):
        all_files += [os.path.join(root, file_name) for file_name in files if file_name.endswith(".wav")]

    train_files = list(set(all_files) - set(validation_files) - set(dev_files))

    # Get the test files
    test_files = list()
    for root, dirs, files in os.walk(os.path.normpath(data_path + "test/")):
        test_files += [os.path.join(root, file_name) for file_name in files if file_name.endswith(".wav")]

    # Get labels
    validation_file_labels = [getLabel(wav) for wav in validation_files]
    dev_file_labels = [getLabel(wav) for wav in dev_files]
    train_file_labels = [getLabel(wav) for wav in train_files]
    test_file_labels = [getLabel(wav) for wav in test_files]

    # Create dictionaries containing (file, labels)
    trainData = {"files": train_files, "labels": train_file_labels}
    valData = {"files": validation_files, "labels": validation_file_labels}
    devData = {"files": dev_files, "labels": dev_file_labels}
    testData = {"files": test_files, "labels": test_file_labels}

    dataDict = {"train": trainData, "val": valData, "dev": devData, "test": testData}

    return dataDict


def getLabel(file_name):
    """
    Extract the label from its file path
    :param file_name: File name
    :return: Class label
    """

    category = os.path.basename(os.path.dirname(file_name))
    label = categories.get(category, categories["_background_noise_"])

    return label


def getDataframe(data, include_unknown=False):
    """
    Create a dataframe from a Dictionary and remove _background_noise_
    :param data: Data dictionary
    :param include_unknown: Whether to include unknown sounds or not
    :return: Dataframe
    """

    df = pd.DataFrame(data)
    df["category"] = df.apply(lambda row: inv_categories[row["labels"]], axis=1)

    if not include_unknown:
        df = df.loc[df["category"] != "_background_noise_", :]

    return df
