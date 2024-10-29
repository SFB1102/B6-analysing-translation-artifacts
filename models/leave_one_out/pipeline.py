import os
import json
import random
import argparse
import pandas as pd
import sys
import ast
from tqdm import tqdm

from evaluate import *
from experiments import *
from load_data import data_prefix_path, get_data

sys.path.append("..")
from classifier.model import XLMRBinaryClassifier


def evaluate_clf(clf, test_data):
    test_texts = [d[0] for d in test_data]
    test_labels = [int(d[1]) for d in test_data]
    predictions = []
    for i in tqdm(range(len(test_texts))):
        input_sent = test_texts[i]
        # true_label = test_labels[i]
        pred_label, _ = clf.predict(input_sent)
        predictions.append(int(pred_label))

    print("Accuracy: ", accuracy(predictions, test_labels))
    print("F1: ", f1(predictions, test_labels))
    print("Precision: ", precision(predictions, test_labels))
    print("Recall: ", recall(predictions, test_labels))


def main_classifier_train():
    root_path = data_prefix_path(DATASET_NAME, "dataset")
    os.makedirs(root_path, exist_ok=True)
    train_path = root_path + "/train.json"
    os.makedirs(data_prefix_path(DATASET_NAME, "main_classifier"), exist_ok=True)
    model_output_path = data_prefix_path(DATASET_NAME, "main_classifier", "model.pt")

    data = get_data(train_path)
    random.shuffle(data)

    train_texts = [d[0] for d in data]
    train_labels = [int(d[1]) for d in data]

    print("Number of 1s in train: ", train_labels.count(1))
    print("Number of 0s in train: ", train_labels.count(0))

    clf = XLMRBinaryClassifier(DATASET_NAME)

    clf.train(train_texts, train_labels, catch_gradient=False)
    clf.save_model(model_output_path)


def main_classifier_evaluate():
    root_path = data_prefix_path(DATASET_NAME, "dataset")
    test_path = root_path + "/test.json"
    model_path = data_prefix_path(DATASET_NAME, "main_classifier", "model.pt")

    clf = XLMRBinaryClassifier(DATASET_NAME)
    clf.load_model(model_path)

    test_data = get_data(test_path)
    evaluate_clf(clf, test_data)


def perform_loo_for_all_sentences(use_train_set=True, start=None, end=None):
    set_name = "train" if use_train_set else "test"

    root_path = data_prefix_path(DATASET_NAME, "dataset")
    sentences_path = root_path + f"/{set_name}.json"

    data = get_data(sentences_path)[start:end]

    model_path = data_prefix_path(DATASET_NAME, "main_classifier", "model.pt")

    clf = XLMRBinaryClassifier(DATASET_NAME)
    clf.load_model(model_path)

    os.makedirs(
        data_prefix_path(DATASET_NAME, "leave_one_out_all_results"), exist_ok=True
    )
    loo_results_path = data_prefix_path(
        DATASET_NAME, "leave_one_out_all_results", f"{set_name}_results.csv"
    )

    leave_one_out_experiment_for_sentences(data, clf, loo_results_path)


def perform_loo_for_sentences_with_1to1_corresp():
    root_path = data_prefix_path(DATASET_NAME, "leave_one_out_1to1_corresp_results")
    os.makedirs(root_path, exist_ok=True)
    df = pd.read_csv(
        os.path.join(root_path, f"{DATASET_NAME}_1to1_corresp_sentences.csv")
    )
    data = []
    for e, row in df.iterrows():
        sentence = row[DATASET_NAME]
        data.append({"sentence": sentence, "label": 1})

    model_path = data_prefix_path(DATASET_NAME, "main_classifier", "model.pt")
    clf = XLMRBinaryClassifier(DATASET_NAME)
    clf.load_model(model_path)
    loo_results_path = data_prefix_path(
        DATASET_NAME,
        "leave_one_out_1to1_corresp_results",
        f"{DATASET_NAME}_leave_one_out_1to1_corresp_results.csv",
    )
    leave_one_out_experiment_for_sentences(
        [(item["sentence"], item["label"]) for item in data], clf, loo_results_path
    )


def extract_top_features_for_loo_all_sentences():
    input_path = os.path.abspath(
        data_prefix_path(
            DATASET_NAME, "leave_one_out_all_results", f"train_results.csv"
        )
    )
    output_path = data_prefix_path(DATASET_NAME, "leave_one_out_all_top_k", "true")
    for k in (1, 3, 5):
        print(f"Extracting top features from dataset '{DATASET_NAME}', k = {k}")
        output = []
        with open(input_path, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter="\t", quotechar='"')
            next(reader)
            for row in tqdm(reader):
                predicted_label, true_label, original_sent, lil = row
                scored_tokens = ast.literal_eval(lil)
                scored_tokens = sorted(
                    scored_tokens, key=lambda p: p[1], reverse=bool(int(true_label))
                )
                top_words = [x[0] for x in scored_tokens]
                result = " ".join(top_words[:k])
                output.append({"sentence": result, "label": int(true_label)})

        os.makedirs(
            os.path.abspath(os.path.join(output_path, f"top_{k}")), exist_ok=True
        )

        with open(
            os.path.abspath(os.path.join(output_path, f"top_{k}", "train.json")), "w"
        ) as fl:
            fl.write("")
            fl.close()

        with open(
            os.path.abspath(os.path.join(output_path, f"top_{k}", "train.json")), "a"
        ) as output_file:
            for s in output:
                output_file.write(json.dumps(s) + "\n")
            output_file.close()


def sufficiency_classifier_train(k):
    os.makedirs(
        data_prefix_path(DATASET_NAME, "leave_one_out_all_top_k", "true"), exist_ok=True
    )
    root_path = data_prefix_path(
        DATASET_NAME, "leave_one_out_all_top_k", "true", f"top_{k}"
    )
    train_path = root_path + "/train.json"
    os.makedirs(
        data_prefix_path(DATASET_NAME, "leave_one_out_sufficiency_classifiers", "true"),
        exist_ok=True,
    )
    model_output_path = data_prefix_path(
        DATASET_NAME, "leave_one_out_sufficiency_classifiers", "true", f"top_{k}.pt"
    )

    data = get_data(train_path)
    random.shuffle(data)

    train_texts = [d[0] for d in data]
    train_labels = [int(d[1]) for d in data]

    print("Number of 1s in train: ", train_labels.count(1))
    print("Number of 0s in train: ", train_labels.count(0))

    clf = XLMRBinaryClassifier(DATASET_NAME)

    clf.train(train_texts, train_labels, catch_gradient=False)
    clf.save_model(model_output_path)
    clf.load_model(model_output_path)


def sufficiency_classifier_evaluate(k):
    root_path = data_prefix_path(DATASET_NAME, "dataset")
    test_path = root_path + "/test.json"

    model_path = data_prefix_path(
        DATASET_NAME, "leave_one_out_sufficiency_classifiers", "true", f"top_{k}.pt"
    )

    clf = XLMRBinaryClassifier(DATASET_NAME)
    clf.load_model(model_path)

    test_data = get_data(test_path)

    evaluate_clf(clf, test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline for integrated gradients (IG) experiments"
    )
    parser.add_argument("--dataset", type=str, help="Dataset to work with")
    parser.add_argument("--action", type=str, help="Action to perform")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--k", type=int, help="top-k tokens")
    args = parser.parse_args()

    DATASET_NAME = args.dataset

    if args.action == "main_classifier_train":
        print("==== Training the main classifier ====")
        main_classifier_train()
        print("==== ====")
    elif args.action == "main_classifier_evaluate":
        print("==== Evaluating the main classifier ====")
        main_classifier_evaluate()
        print("==== ====")
    elif args.action == "perform_loo_for_all_sentences":
        print("==== Performing Leave-One-Out (LOO) on TRAIN set for all sentences ====")
        perform_loo_for_all_sentences(
            use_train_set=True, start=args.start, end=args.end
        )
        print("==== ====")
    elif args.action == "perform_loo_for_sentences_with_1to1_corresp":
        print(
            (
                "==== Performing Leave-One-Out (LOO) on sentences with 1-to-1 correspondence "
                "between original (source) and translation (target) ===="
            )
        )
        perform_loo_for_sentences_with_1to1_corresp(DATASET_NAME)
        print("==== ====")
    elif args.action == "extract_top_features_for_loo_all_sentences":
        print(f"==== Extracting top features for LOO performed on all sentences ====")
        extract_top_features_for_loo_all_sentences()
    elif args.action == "sufficiency_classifier_train":
        print(f"= Training the sufficiency classifier using TRUE labels, k={args.k} =")
        sufficiency_classifier_train(args.k, use_predicted_labels=False)
        print("==== ====")
    elif args.action == "sufficiency_classifier_evaluate":
        print(
            f"= Evaluating on test set of whole sentences with TRUE labels, k={args.k} ="
        )
        sufficiency_classifier_evaluate(args.k, use_predicted_labels=False)
        print("==== ====")
    else:
        raise ValueError("No such action")
