# DSC180A-code

# Overview
Code for the baseline models implemented for five tasks across two datasets: NLI on the SNLI dataset, poltical affiliation classification on the LIAR dataset, and veracity prediction on the LIAR dataset. I'll revisit all these tasks by adding complexity to my baseline models (and possibly using different datasets). The other two tasks are a multi-factor approach to label classification on the LIAR dataset (using an ensemble which aggregates my other models, in the `overall_model.ipynb` notebook and a NLI model using naive evidence retrieval (based on named entity recognition and the wikipedia api). The most successful six-way classification on LIAR is in the `LIAR Sota replication` notebook, which is based on the state of the art model on the LIAR dataset.

## Data
- SNLI: Available through the [Stanford NLP](https://nlp.stanford.edu/projects/snli/) website. The archive can be downloaded through the "download" link and unzipped in the same directory as my SNLI_notebook.ipynb to facilitate its execution.
- LIAR: Available at [HuggingFace](https://huggingface.co/datasets/liar), among other sources. This dataset can be downloaded, and all files can be placed in the same directory as my clean_liar_bias_classification.ipynb and liar_party_classification.ipynb files.
- LIAR plus: Extension of the LIAR dataset available [here](https://github.com/Tariq60/LIAR-PLUS)

## Packages
- All code is implemented in either PyTorch or TensorFlow/Keras. Standard data science/ML packages (Pandas, NumPy, and sk-Learn) and NLP packages (NLTK and spaCy) are also employed.
- The [Wikipedia API](https://pypi.org/project/Wikipedia-API/) is used for naive evidence retrieval
- Code is implemented in Jupyter Notebooks, which can be run sequentially to reproduce the results.
- Pretrained models are provided by HuggingFace through the [transformers package](https://huggingface.co/docs/transformers/index)
