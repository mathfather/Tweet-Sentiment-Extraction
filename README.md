# Tweet-Sentiment-Extraction
In this project we use both recurrent neural models and Transformer-based neural models to attempt the Tweet-Sentiment-Extraction task from [here](https://www.kaggle.com/c/tweet-sentiment-extraction/overview/description).
## Project Overview
- The dataset can be accessed from **data** folder. The **cleaned_train.csv** is the truncated version of our training set(Drop the 5 longest sentences of the original dataset).
- We're trying to predict the *selected text* of each example, where the excerpt from the original tweet should best support the sentiment characterization.
- The **figs** folder contains all the plots that we used in the report paper. It provides two both 'svg' and 'png' versions of the plots.
- Folder **logs** includes the jaccard score records for experiments.
- Folder **src** includes all the significant files for training our models. The file `recurrent_models.py` includes the training file for reucrrent neural models modified from a colab notebook. The `starter.py` includes the necessary scripts for ploting. The rest of files are for different parts of the BERT models.
- Our report paper can be found [here](https://github.com/mathfather/Tweet-Sentiment-Extraction/blob/master/report.pdf).
## Requirements

### Recurrent neural models

- used GloVe.Twitter 25d file for embedding
- tensorflow == 2.2.0

### BERT models

- pytorch == 1.2.0
- numpy == 1.18.3
- pandas == 0.25.3
- tqdm == 4.45.0
- sklearn == 0.22.1
- transformers == 2.8.0
- tokenizers == 0.5.2

## Experiments reproduction

### Recurrent neural models

To run the best recurrent model, run `python3 recurrent_models.py -i train_file_path -em embedding_file_path`. Other implementation details can also be found in the `recurrent_models.py` script.

### BERT models

To reproduce BERT model experiments, you can directly run `BertBaseWithOutSentiment.py` and `BertBaseWithSentiment.py` under the folder **src**. Some hpyerparameters can be changed, and the way of making modifications is indicated in the scripts.
