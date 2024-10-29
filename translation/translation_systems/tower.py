import torch
from transformers import pipeline
import re
from math import ceil, floor
from tqdm import tqdm
import json
import argparse

from transformers.utils import logging

logging.get_logger("transformers").setLevel(logging.ERROR)


pipe = pipeline(
    "text-generation",
    model="Unbabel/TowerInstruct-v0.2",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


def process(output):
    return output


def translate(source_text: str):
    messages = [
        {
            "role": "user",
            "content": (
                "Translate the following text from German into English.\n"
                + f"German: {source_text}\nEnglish:"
            ),
        }
    ]
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    outputs = pipe(prompt, max_new_tokens=256, do_sample=False)
    return process(outputs[0]["generated_text"])


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
    parser = argparse.ArgumentParser(description="Run tower translation")
    parser.add_argument("--source_file", type=str, help="Source file path")
    parser.add_argument("--output_file", type=str, help="Output file path")
    args = parser.parse_args()
    process_sources(args.sources_file, args.output_file)
