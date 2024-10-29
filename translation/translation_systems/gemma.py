import re
import datasets
from transformers.pipelines.pt_utils import KeyDataset
import torch
import json
import argparse
import transformers
from tqdm import tqdm

datasets.logging.set_verbosity(datasets.logging.DEBUG)

batch_size = 16

pipeline = transformers.pipeline(
    "text-generation",
    model="google/gemma-7b",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    truncation=True,
    padding=True,
    batch_size=batch_size,
)
pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id  # [0]


def sentences_to_dataset(sentences):
    data = {
        "TD": list(range(len(sentences))),
        "text": [
            (
                "Translate this sentence from German to English without any comments:\nGerman: "
                f"{text}\nEnglish:"
            )
            for text in sentences
        ],
    }
    return datasets.Dataset.from_dict(data)


def parse(text):
    try:
        text = re.sub(r"to\s+English|<\/?[be]os>", "", text)
        return re.search(r"English:[\s\n]*(.+)", text).group(1)
    except:
        return text


def process_sources(source_file_name, output_file_name):
    with open(source_file_name, "r", encoding="utf-8") as f:
        sentences = f.read().splitlines()
        print("Making dataset...")
        chunk_size = 128
        print("Running pipeline...")
        for i in tqdm(range(0, len(sentences), chunk_size)):
            torch.cuda.empty_cache()
            sents = sentences[i : i + chunk_size]
            dataset = sentences_to_dataset(sents)
            for e, out in enumerate(
                pipeline(KeyDataset(dataset, "text"), max_new_tokens=128)
            ):
                with open(output_file_name, "a") as ff:
                    ff.write(
                        json.dumps(
                            {
                                "index": i + e,
                                "translation": parse(out[0]["generated_text"]),
                            }
                        )
                        + "\n"
                    )
                    ff.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run gemma translation")
    parser.add_argument("--source_file", type=str, help="Source file path")
    parser.add_argument("--output_file", type=str, help="Output file path")
    args = parser.parse_args()

    process_sources(args.sources_file, args.output_file)
