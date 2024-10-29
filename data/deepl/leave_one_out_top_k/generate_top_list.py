import os
from string import punctuation
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
import ast
import csv
from pathlib import Path


def generate_top(set_type, k = 1, use_predicted_labels = True, prefix=""):
    assert set_type in ("train", "test")
    if prefix:
        prefix += "_"
    loo_result_path = os.path.abspath(
        os.path.join(
            os.path.abspath(__file__),
            "..", "..", "loo_results", f"{prefix}{set_type}_results.csv"
        )
    )
    output = []
    with open(loo_result_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='"')
        next(reader)
        for row in tqdm(reader):
            predicted_label, true_label, original_sent, scored_tokens = row
            top_words = [x[0] for x in ast.literal_eval(scored_tokens)]
            result = " ".join(top_words[:k])
            if use_predicted_labels:
                output.append({"sentence": result, "label": int(predicted_label)})
            else:
                output.append({"sentence": result, "label": int(true_label)})

    folder_name = "predicted" if use_predicted_labels else "true"

    Path(os.path.join(".", folder_name, f"top_{k}")).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(".", folder_name, f"top_{k}", f"{prefix}{set_type}.json"), "w") as fl:
        fl.write("")
        fl.close()

    with open(os.path.join(".", folder_name, f"top_{k}", f"{prefix}{set_type}.json"), "a") as output_file:
        for s in output:
            output_file.write(json.dumps(s) + "\n")
        output_file.close()


def generate(prefix=None):
    for k in (1, 3, 5):
        for st in ("train",):# "test"):
            for label_type in ("true",):
                print(f"Generating for k = {k}, '{st}' set type, '{label_type}' label type")
                use_predicted_labels = (label_type == "predicted")
                generate_top(st, k = k, use_predicted_labels = use_predicted_labels, prefix = prefix)

if __name__ == "__main__":
    generate(prefix="")
