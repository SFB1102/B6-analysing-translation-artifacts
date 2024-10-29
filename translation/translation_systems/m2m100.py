from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from tqdm import tqdm
import json
import argparse

from transformers.utils import logging

logging.get_logger("transformers").setLevel(logging.ERROR)


model = M2M100ForConditionalGeneration.from_pretrained(
    "facebook/m2m100_418M", device_map="auto"
)
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")


def translate(source_text: str):
    tokenizer.src_lang = "de"
    encoded_de = tokenizer(source_text, return_tensors="pt").to(model.device)
    generated_tokens = model.generate(
        **encoded_de, forced_bos_token_id=tokenizer.get_lang_id("en")
    )
    resp = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return resp[0]


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
    parser = argparse.ArgumentParser(description="Run m2m100 translation")
    parser.add_argument("--source_file", type=str, help="Source file path")
    parser.add_argument("--output_file", type=str, help="Output file path")
    args = parser.parse_args()

    process_sources(args.sources_file, args.output_file)
