from evaluate import *
from tqdm import tqdm
import csv


def leave_one_out_experiment_per_sent(original_sent, model):
    original_pred_label, original_proba = model.predict(original_sent)
    print(f"Original sentence: {original_sent}")
    print(f"Original sentence prediction: {original_pred_label}")
    print(f"Original sentence probability: {original_proba}")
    print("\n")

    words = original_sent.split()

    for i in range(len(words)):
        left_out_word = words.pop(i)
        left_out_sentence = " ".join(words)

        llo_pred_label, llo_proba = model.predict(left_out_sentence)
        print(f"Left out word: {left_out_word}")
        print(f"Left out sentence: {left_out_sentence}")
        print(f"Left out sentence prediction: {llo_pred_label}")
        print(f"Left out sentence probability: {llo_proba}")
        print("\n")

        words.insert(i, left_out_word)


def leave_one_out_experiment_for_sentences(data, model, result_file_path):
    predictions = []
    labels = []
    results = []

    with open(result_file_path, "w", newline="", encoding="utf-8") as result_file:
        writer = csv.writer(result_file, delimiter="\t")
        writer.writerow(
            ["predicted_labels", "true_labels", "original_sent", "lil_interpretations"]
        )
        for original_sent, original_label, ix in tqdm(data):
            labels.append(int(original_label))
            original_pred_label, original_proba = model.predict(original_sent)
            predictions.append(int(original_pred_label))

            words = original_sent.split()
            prob_diffs = {}

            for i in range(len(words)):
                left_out_word = words.pop(i)
                left_out_sentence = " ".join(words)

                llo_pred_label, llo_proba = model.predict(left_out_sentence)
                prob_diff = round((abs(original_proba - llo_proba) * 10000), 4)

                if (
                    left_out_word not in prob_diffs
                    or prob_diff > prob_diffs[left_out_word]
                ):
                    prob_diffs[left_out_word] = prob_diff

                words.insert(i, left_out_word)

            top_words = sorted(prob_diffs.items(), key=lambda x: x[1], reverse=True)
            writer.writerow(
                [original_pred_label, original_label, original_sent, top_words]
            )
            results.append(
                [original_pred_label, original_label, original_sent, top_words]
            )

        print("Accuracy: ", accuracy(predictions, labels))
        print("F1: ", f1(predictions, labels))
        print("Precision: ", precision(predictions, labels))
        print("Recall: ", recall(predictions, labels))
