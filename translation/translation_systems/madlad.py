from tqdm import tqdm
import json
import argparse
from transformers import T5ForConditionalGeneration, T5Tokenizer

model_name = "jbochi/madlad400-3b-mt"
model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
tokenizer = T5Tokenizer.from_pretrained(model_name)


def translate(source_text: str):
    text = f"<2en> {source_text}"
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(input_ids=input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


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
    parser = argparse.ArgumentParser(description="Run madlad translation")
    parser.add_argument("--source_file", type=str, help="Source file path")
    parser.add_argument("--output_file", type=str, help="Output file path")
    args = parser.parse_args()

    process_sources(args.sources_file, args.output_file)
