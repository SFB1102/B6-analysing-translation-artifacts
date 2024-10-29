import transformers
import torch
from math import ceil, floor
from tqdm import tqdm
import json
import argparse
from transformers.pipelines.pt_utils import KeyDataset
import datasets
import time

datasets.logging.set_verbosity(datasets.logging.DEBUG)

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    truncation=True,
    batch_size=64,
)
pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id[0]


def sentences_to_dataset(sentences):
    data = {
        "TD": list(range(len(sentences))),
        "msg": [
            [
                {
                    "role": "system",
                    "content": "Translate the following sentence from German to English:",
                },
                {"role": "user", "content": text + "\nEnglish:"},
            ]
            for text in sentences
        ],
    }
    return datasets.Dataset.from_dict(data)


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
                pipeline(KeyDataset(dataset, "msg"), max_new_tokens=256)
            ):
                with open(output_file_name, "a") as fl:
                    fl.write(
                        json.dumps(
                            {
                                "index": i + e,
                                "translation": out[0]["generated_text"][-1]["content"],
                            }
                        )
                        + "\n"
                    )
                    fl.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run llama-31-it-8b translation")
    parser.add_argument("--source_file", type=str, help="Source file path")
    parser.add_argument("--output_file", type=str, help="Output file path")
    args = parser.parse_args()

    process_sources(args.sources_file, args.output_file)
