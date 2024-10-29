import argparse
import logging as log
import os
import pandas as pd
import json
import numpy as np
import ast
import torch
import itertools
import csv
from load_data import data_prefix_path, get_data

import sys

sys.path.append("..")
from classifier.model import XLMRBinaryClassifier

from argparse import RawTextHelpFormatter
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz
from tqdm import tqdm
from model_loader import (
    generate_tensors_for_ig,
    tokenize,
    summarize_attributions,
    load_model,
)
from evaluate import *


gpu_device = torch.device("cuda")
cpu_device = torch.device("cpu")


def custom_forward(model):
    def fn(input_tensor):
        """Custom forward that selects only logits from the output
        of the actual forward"""
        model_output = model(input_tensor)
        return model_output.logits

    return fn


def process_scored_tokens(scored_tokens):
    agg_func_pieces = np.mean
    agg_func_identical = max
    res = []
    for tok in scored_tokens:
        token, score = tok
        if token in ("<s>", "</s>"):
            continue
        if token.startswith("▁"):
            res.append([token, [score]])
        else:
            res[-1][0] += token
            res[-1][1].append(score)
    concatenated = []
    for tok in res:
        token = tok[0]
        assert token.startswith("▁")
        token = token[1:]
        scores = tok[1]
        concatenated.append((token, agg_func_pieces(scores)))
    concatenated.sort(key=lambda x: x[0])
    deduplicated = {
        token: agg_func_identical(map(lambda k: k[1], group))
        for token, group in itertools.groupby(concatenated, key=lambda x: x[0])
    }

    return sorted(deduplicated.items(), key=lambda p: p[1], reverse=True)


def compute_attribution_scores_1to1_sentences():
    loo_1to1_path = data_prefix_path(DATASET_NAME, "leave_one_out_1to1_corresp_results")
    root_path = data_prefix_path(DATASET_NAME, "integrated_grads_1to1_corresp_results")
    os.makedirs(root_path, exist_ok=True)

    result_file_path = os.path.join(
        root_path, f"{DATASET_NAME}_integrated_grads_1to1_corresp_results.csv"
    )

    df = pd.read_csv(
        os.path.join(loo_1to1_path, f"{DATASET_NAME}_1to1_corresp_sentences.csv")
    )
    data = []
    for e, row in df.iterrows():
        sentence = row[DATASET_NAME]
        data.append({"sentence": sentence, "label": 1})
    model, tokenizer = load_model()

    model.eval()
    model.zero_grad()

    lig = LayerIntegratedGradients(custom_forward(model), model.roberta.embeddings)

    os.makedirs(data_prefix_path(DATASET_NAME, "dataset"), exist_ok=True)

    with open(result_file_path, "w", newline="", encoding="utf-8") as result_file:
        writer = csv.writer(result_file, delimiter="\t")
        writer.writerow(["true_label", "original_sent", "scored_tokens"])
        for line in tqdm(data):
            # line = json.loads(line)
            text = line["sentence"]
            label = line["label"]
            gt_class = int(label)

            input_tensor, ref_tensor = generate_tensors_for_ig(
                text, tokenizer, max_len=240
            )

            tokens = tokenize(text, tokenizer, max_len=512)

            attributions = lig.attribute(
                inputs=input_tensor, baselines=ref_tensor, target=gt_class
            )
            attributions = summarize_attributions(attributions)
            attributions = attributions.cpu().numpy()

            scored_tokens = []
            for token, attr in zip(tokens, attributions):
                scored_tokens.append((token, attr))
            scored_tokens = process_scored_tokens(scored_tokens)
            scored_tokens = [(w, float(s)) for (w, s) in scored_tokens]
            writer.writerow([gt_class, text, str(scored_tokens)])


def compute_attribution_scores_for_all_sentences():
    model, tokenizer = load_model()

    model.eval()
    model.zero_grad()

    lig = LayerIntegratedGradients(custom_forward(model), model.roberta.embeddings)

    os.makedirs(data_prefix_path(DATASET_NAME, "dataset"), exist_ok=True)
    dataset_file = open(data_prefix_path(DATASET_NAME, "dataset", "train.json"))

    os.makedirs(
        data_prefix_path(DATASET_NAME, "integrated_grads_all_results"), exist_ok=True
    )
    result_file_path = open(
        data_prefix_path(
            DATASET_NAME,
            "integrated_grads_all_results",
            f"{DATASET_NAME}_integrated_grads_all_results.csv",
        )
    )

    with open(result_file_path, "w", newline="", encoding="utf-8") as result_file:
        writer = csv.writer(result_file, delimiter="\t")
        writer.writerow(["true_label", "original_sent", "scored_tokens"])
        for line in tqdm(dataset_file.readlines()):
            line = json.loads(line)
            text = line["sentence"]
            label = line["label"]
            gt_class = int(label)

            input_tensor, ref_tensor = generate_tensors_for_ig(
                text, tokenizer, max_len=240
            )

            tokens = tokenize(text, tokenizer, max_len=512)

            attributions = lig.attribute(
                inputs=input_tensor, baselines=ref_tensor, target=gt_class
            )
            attributions = summarize_attributions(attributions)
            attributions = attributions.cpu().numpy()

            scored_tokens = []
            for token, attr in zip(tokens, attributions):
                scored_tokens.append((token, attr))
            scored_tokens = process_scored_tokens(scored_tokens)
            scored_tokens = [(t, float(a)) for (t, a) in scored_tokens]
            writer.writerow([gt_class, text, str(scored_tokens)])


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


def extract_top_features_all_sentences():
    input_path = os.path.abspath(
        data_prefix_path(
            DATASET_NAME,
            "integrated_grads_all_results",
            f"{DATASET_NAME}_integrated_grads_all_results.csv",
        )
    )
    output_path = data_prefix_path(DATASET_NAME, "integrated_grads_top_k", "true")
    os.makedirs(output_path, exist_ok=True)
    for k in (1, 3, 5):
        print(f"Extracting top features from dataset '{DATASET_NAME}', k = {k}")
        output = []
        with open(input_path, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter="\t", quotechar='"')
            next(reader)
            for row in tqdm(reader):
                true_label, original_sent, tokens = row
                scored_tokens = ast.literal_eval(tokens)

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
        data_prefix_path(DATASET_NAME, "integrated_grads_top_k", "true"), exist_ok=True
    )
    root_path = data_prefix_path(
        DATASET_NAME, "integrated_grads_top_k", "true", f"top_{k}"
    )
    train_path = root_path + "/train.json"

    os.makedirs(
        data_prefix_path(
            DATASET_NAME, "integrated_grads_sufficiency_classifiers", "true"
        ),
        exist_ok=True,
    )
    model_output_path = data_prefix_path(
        DATASET_NAME, "integrated_grads_sufficiency_classifiers", "true", f"top_{k}.pt"
    )

    data = get_data(train_path)

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
        DATASET_NAME, "integrated_grads_sufficiency_classifiers", "true", f"top_{k}.pt"
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
    parser.add_argument("--k", type=int, help="top-k tokens")
    args = parser.parse_args()
    DATASET_NAME = args.dataset

    if args.action == "compute_attribution_scores_for_all_sentences":
        print("==== Computing attribution scores for all sentences ====")
        compute_attribution_scores_for_all_sentences()
    elif args.action == "compute_attribution_scores_1to1_corresp_sentences":
        print(
            "==== Computing attribution scores for sentences with 1-to-1 source-target correspondence ===="
        )
        compute_attribution_scores_1to1_sentences()
    elif args.action == "extract_top_features_for_all_sentences":
        print(f"==== Extracting top features for all sentences ====")
        extract_top_features_all_sentences()
    elif args.action == "sufficiency_classifier_train":
        print(f"==== Training suff. classifiers using TRUE labels, k={args.k} ====")
        sufficiency_classifier_train(args.k)
    elif args.action == "sufficiency_classifier_evaluate":
        print(f"= Evaluating on test set of top-k TRUE, k={args.k} =")
        sufficiency_classifier_evaluate(args.k)
    else:
        raise ValueError("No such action")
