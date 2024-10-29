import argparse
import math
import os
import pandas as pd
import torch
import transformers
from enum import Enum

from argparse import RawTextHelpFormatter
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz
from transformers import BertTokenizer, BertForSequenceClassification

# cuda_device=0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    model_name = "xlm-roberta-base"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    return model, tokenizer


def summarize_attributions(attributions):
    """Find a mean attribution value for each embedding"""
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


def generate_tensors_for_ig(paragraph, tokenizer, max_len=math.inf):
    """Generate a paragraph tensor and a reference tensor

    Args:
        paragraph (str): stripped paragraph in the original case
        tokenizer (transformers.BertTokenizer)
        max_len (int): maximal number of tokens; if it is surpassed,
                  the sequence will be cut to meet the required length

    Returns:
        input_tensor (torch.LongTensor)
        ref_tensor (torch.LongTensor): pad tokens of the same shape
             as the original paragraph
    """
    input_ids = tokenizer.encode(paragraph)
    input_len = len(input_ids)
    if input_len > max_len:
        input_len = max_len
        input_ids = input_ids[: (max_len - 1)] + [tokenizer.sep_token_id]
    ref_input_ids = (
        [tokenizer.cls_token_id]
        + [tokenizer.pad_token_id] * (input_len - 2)
        + [tokenizer.sep_token_id]
    )

    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    ref_tensor = torch.tensor(ref_input_ids, dtype=torch.long).unsqueeze(0)

    input_tensor = input_tensor.to(device)
    ref_tensor = ref_tensor.to(device)
    return input_tensor, ref_tensor


def tokenize(paragraph, tokenizer, max_len=math.inf):
    """Tokenize the paragraph

    Args:
        paragraph (str): stripped paragraph in the original case
        tokenizer (transformers.BertTokenizer): tokenizer
        max_len (int): maximal number of tokens; if it is surpassed,
                  the sequence will be cut to meet the required length

    Returns:
        tokens (list of str)?
    """

    input_ids = tokenizer.encode(paragraph)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    if len(tokens) > max_len:
        tokens = tokens[: (max_len - 1)] + [tokenizer.sep_token]
    return tokens
