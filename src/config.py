from tokenizers import BertWordPieceTokenizer
import os

MAX_LEN = 64
VOCAB = "vocab.txt"
CONFIG = "config.json"
MODEL = "pytorch_model.bin"
BERT_BASE = "./models/bert_base"
TRAIN_SET = "./data/train.csv"
TEST_SET = "./data/test.csv"
CLEANED_TRAIN_SET = "./data/cleaned_train.csv"
BERT_BASE_TOKENIZER = BertWordPieceTokenizer(
    os.path.join(BERT_BASE, VOCAB),
    lowercase=True
)
CHECKPOINT_FOLDER = "./checkpoints"
BERTBASEWITHOUTSENTIMENT = "BertBaseWithOutSentiment"
BERTBASEWITHSENTIMENT = "BertBaseWithSentiment"

SENT_DIS = "./figs/sent_dis.png"
SENTIMENT_DIS = "./figs/sentiment_dis.png"
SENTIMENT_CONDITION_DIS = "./figs/sentiment_condition_dis.png"
LOG_FOLDER = "./logs"