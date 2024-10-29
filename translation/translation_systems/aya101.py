from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import json
import argparse

from transformers.utils import logging

logging.get_logger("transformers").setLevel(logging.ERROR)


checkpoint = "CohereForAI/aya-101"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, device_map="auto")


def translate(source_text: str):
    inputs = tokenizer.encode(
        f"Translate to English: {source_text}", return_tensors="pt"
    ).to(model.device)
    outputs = model.generate(inputs, max_new_tokens=128)
    return tokenizer.decode(outputs[0])


def process_sources(source_file_name, output_file_name):
    with open(source_file_name, "r", encoding="utf-8") as f:
        sentences = f.read().splitlines()
        for e in tqdm(range(len(sentences))):
            sentence = sentences[e]
            translation = translate(sentence)
            with open(output_file_name, "a") as fl:
                fl.write(json.dumps({"index": e, "translation": translation}) + "\n")
                fl.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run aya101 translation")
    parser.add_argument("--source_file", type=str, help="Source file path")
    parser.add_argument("--output_file", type=str, help="Output file path")
    args = parser.parse_args()

    process_sources(args.sources_file, args.output_file)
