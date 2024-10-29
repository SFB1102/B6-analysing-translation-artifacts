This is the official repository for "Analysing Translation Artifacts: A Comparative Study of LLMs, NMTs, and Human Translations". We provide the code necessary to reproduce the experiments described in the paper, along with the pre-processed datasets derived from Europarl data. We also include the code for the notebooks used to generate the figures presented in the paper.

# Requirements
`pip install -r requirements.txt`

# Data
In our paper, we considered the following translation systems, along with the human translation data:
* DeepL
* Google Translte
* M2M-100-418M
* MADLAD-400-MT
* NLLB-600M
* LLaMAX-3.1-8B-Alpaca
* TowerInstruct-7B-v0.2
* Aya-101-13B
* Gemma-7B
* Llama-3.1-IT-8B

The derived datasets you can find in `data/` directory are originally based on Europarl data, which were translated by the mentioned systems. In the `data/` directory for each translation system, the pre-processed dataset for training the linear classifier is available, along with results from the Leave-One-Out (LOO) and Integrated Gradients (IG) experiments.

In case of open systems, we also provide the code used to retrieve the translations. This code as well as the COMET/BLEU evaluation scripts can be found in `translation/` folder.

# Training & evaluation
Code for training and evaluating the linear classifier and subsequent Leave-One-Out (LOO) and Integrated Gradients (IG) experiments can be found in the `models/` directory; `graphs/` folder contains code related to the important visualizations.
