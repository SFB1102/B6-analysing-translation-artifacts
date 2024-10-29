import fasttext
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from tqdm import tqdm
import csv
import json
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


def load_fasttext_model(model_path):
    return fasttext.load_model(model_path)


def load_translation_pipeline(checkpoint, src_lang, tgt_lang):
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        max_length=400,
        device=0,
    )


fasttext_model_path = "lid218e.bin"
fasttext_model = load_fasttext_model(fasttext_model_path)

translation_checkpoint = "facebook/nllb-200-distilled-600M"
src_lang = "deu_Latn"
tgt_lang = "eng_Latn"
translation_pipeline = load_translation_pipeline(
    translation_checkpoint, src_lang, tgt_lang
)


def read_text_file(file_path):
    return open(file_path).splitlines()


def translate(source_text: str):
    return translation_pipeline(source_text)[0]["translation_text"]


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
    parser = argparse.ArgumentParser(description="Run nllb translation")
    parser.add_argument("--source_file", type=str, help="Source file path")
    parser.add_argument("--output_file", type=str, help="Output file path")
    args = parser.parse_args()

    process_sources(args.sources_file, args.output_file)
