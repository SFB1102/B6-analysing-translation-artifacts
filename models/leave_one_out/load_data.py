import json
import os


def get_data(path):
    with open(path, "r", encoding="utf-8") as input_file:
        data = []
        for k, line in enumerate(input_file):
            json_line = json.loads(line)
            sentence = json_line["sentence"]
            label = json_line["label"]
            data.append((sentence, label, k))
    return data


def data_prefix_path(*path):
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", *path
    )
