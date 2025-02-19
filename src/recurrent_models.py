# -*- coding: utf-8 -*-
"""tweet_sentiment_extract

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FHqQBzqxjRWwPNp6dQk27c0syFEDNqpi
"""

# -*- coding: utf-8 -*-
import torch
import pandas as pd
import json
import sys
import nltk
import re
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from keras.models import Model
from keras.layers import Embedding, LSTM, Dropout,Dense, Bidirectional, dot, Input, concatenate
from keras import optimizers
from keras import backend as K
# !pip install transformers
from transformers import *
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,DistributedSampler
from torch.utils.data import TensorDataset, random_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import argparse
import random
# nltk.download('stopwords')
random.seed(233)
STOPWORDS = set(stopwords.words('english'))
args = argparse.ArgumentParser(description='Program description.')
args.add_argument('-i','--input-file', type=str, help='Input file', default='../data/train.csv')
args.add_argument('-t','--test-file', type=str, help='Test file', default='../data/test.csv')
args.add_argument('-em','--embedding-file', type=str, help='Embedding file', default='../embedding/glove.twitter.27B.25d.txt')
args = args.parse_args()

# original file run on Google Colab
train_file_path = args.input_file
test_file_path = args.test_file
embedding_file_path = args.embedding_file

train_facts = pd.read_csv(train_file_path)
test_facts = pd.read_csv(test_file_path)

train_facts_drop = train_facts.dropna(axis=0)
test_facts = test_facts.dropna(axis=0)

# print('train dataset length: ', len(train_facts_drop))
# print('test dataset length: ', len(test_facts))

# print('train file: ', train_facts.head())
# print('train info: ', train_facts.info()) # selected text is the gold label
OOV = '<OOV>'
PAD = '<PAD>'
max_length=150
vocab_size=10000


train_facts_drop = train_facts_drop.reset_index(drop=True)
# get rid of extra spaces for train and test
for ind in train_facts_drop.index:
  train_facts_drop['sentiment'][ind] = train_facts_drop['sentiment'][ind].strip()
  train_facts_drop['text'][ind] = train_facts_drop['text'][ind].strip()
for ind in test_facts.index:
  test_facts['sentiment'][ind] = test_facts['sentiment'][ind].strip()  
  test_facts['text'][ind] = test_facts['text'][ind].strip()

# concatenate the sentiment and the text into pseudo-sentences
cols = ['sentiment','text']
train_facts_drop['comb'] = train_facts_drop[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
train_comb_texts = train_facts_drop['comb'].tolist()


print('length of all train texts: ', len(train_facts_drop))
# print('length of all test texts: ', len(test_comb_texts))
# print('Example combined sentences: ', test_comb_texts[:5])

# no textual preprocessing needed, because you'll be comparing the textual similarity
# for example no case normalization
# get the start and end indices, make gold labels
def get_start_end_list(df, column):
  # print('start and end positions are found in the concatenated sentiment+text if column=='comb', else found in text')
  start = []
  end = []
  for ind in range(len(df)):
    match = re.search(re.escape(df['selected_text'][ind]), df[column][ind])
    start.append(match.start())
    end.append(match.end())
  assert len(start) == len(end)
  return start, end

# depending on the column the indices will be different
start, end = get_start_end_list(train_facts_drop, 'comb')
start_in_text, end_in_text = get_start_end_list(train_facts_drop, 'text')

# indices sanity check
assert train_facts_drop['text'][23876][start_in_text[23876]:end_in_text[23876]] == train_facts_drop['selected_text'][23876]

# split into train and validation

# mode = concatenate tweet and sentiment as one input
train_size = int(len(train_facts_drop) * 0.85)
train_size = 23360
all_texts = train_comb_texts
# print('example text: ', all_texts[0])
# print('example selected text: ', train_facts_drop['selected_text'][0])
# print('example start and end: ', start[0], end[0])
train_texts = all_texts[0: train_size]
train_start = start[:train_size]
train_end = end[:train_size]

validation_texts = all_texts[train_size:]
validation_start = start[train_size:]
validation_end = end[train_size:]


tokenizer = Tokenizer( oov_token=OOV, lower=False) # filters=

tokenizer.fit_on_texts(train_texts)
word_index = tokenizer.word_index 
print('length of vocab: ', len(word_index))
# print('OOV index: ',word_index[OOV])

# Turn each text into a sequence of integer word IDs
train_sequences = tokenizer.texts_to_sequences(train_texts)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')

# more OOV items in the dev set
validation_sequences = tokenizer.texts_to_sequences(validation_texts)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding='post', truncating='post')

print('train padded shape: ', train_padded.shape)
# print('train_padded example ', train_padded[0])
print('val padded shape: ', validation_padded.shape)


# mode = separate inputs
# =========================================================================================
# inputs are train_padded_tweet, train_sentiment_oh, || val_padded_tweet, val_sentiment_oh
#   labels: start/end_in_text, texts are train_tweets and val_tweets
# ==========================================================================================
train_padded_tweet = [sent[1:] for sent in train_padded]
train_padded_tweet = np.array(train_padded_tweet) # shape = (23358, 149)

validation_padded_tweet = [sent[1:] for sent in validation_padded]
val_padded_tweet = np.array(validation_padded_tweet)

train_sentiment = [text.split()[0] for text in train_texts] # list of strings
val_sentiment = [text.split()[0] for text in validation_texts]



all_tweets = train_facts_drop['text'].values.tolist()
train_tweets = all_tweets[:train_size] # not tokenized, list of strings
val_tweets = all_tweets[train_size:] # not tokenized, list of strings


# one-hot encode sentiment:
# positive [1,0,0] neutral=[0,1,0], negative=[0,0,1]; for auxiliary input
train_sentiment_oh = np.zeros((len(train_sentiment), 3))
for i, senti in enumerate(train_sentiment):
  if senti=='positive': 
    train_sentiment_oh[i,0] = 1
  elif senti =='neutral':
    train_sentiment_oh[i,1] = 1
  else:
    train_sentiment_oh[i,2] =1

# one hot encode sentiment for validation set
val_sentiment_oh = np.zeros((len(val_sentiment), 3))
for i, senti in enumerate(val_sentiment):
  if senti=='positive': 
    val_sentiment_oh[i,0] = 1
  elif senti =='neutral':
    val_sentiment_oh[i,1] = 1
  else:
    val_sentiment_oh[i,2] =1

# get gold label indices
train_start_in_text = start_in_text[:train_size]
train_end_in_text = end_in_text[:train_size]

validation_start_in_text = start_in_text[train_size:]
validation_end_in_text = end_in_text[train_size:]

assert len(validation_start_in_text) == len(validation_end_in_text)
assert len(validation_start_in_text) == len(val_tweets)
assert len(train_start_in_text) == len(train_tweets)

assert val_tweets[516][validation_start_in_text[516]:validation_end_in_text[516]] \
       == train_facts_drop['selected_text'][23876]

# one-hot encode the start and end indices
train_label_hot = np.zeros(train_padded.shape)
train_start_hot = np.zeros(train_padded.shape)
train_end_hot = np.zeros(train_padded.shape)

train_start_in_text_hot = np.zeros(train_padded_tweet.shape)
train_end_in_text_hot = np.zeros(train_padded_tweet.shape)

for i in range(train_label_hot.shape[0]):
  train_label_hot[i, train_start[i]] = 1
  train_label_hot[i, train_end[i]] = 1

  train_start_hot[i, train_start[i]] = 1
  train_end_hot[i, train_end[i]] = 1
  train_start_in_text_hot[i,train_start_in_text[i]] = 1
  train_end_in_text_hot[i, train_end_in_text[i]] = 1

assert train_label_hot[32, train_start[32]] ==1 


#======= do the same for val ==============
val_label_hot = np.zeros(validation_padded.shape)
val_start_hot = np.zeros(validation_padded.shape)
val_end_hot = np.zeros(validation_padded.shape)

val_start_in_text_hot = np.zeros(val_padded_tweet.shape)
val_end_in_text_hot = np.zeros(val_padded_tweet.shape)

for i in range(val_label_hot.shape[0]):
  val_label_hot[i, validation_start[i]] = 1
  val_label_hot[i, validation_end[i]] = 1

  val_start_hot[i, validation_start[i]] = 1
  val_end_hot[i, validation_end[i]] = 1
  val_start_in_text_hot[i, validation_start_in_text[i]] = 1
  val_end_in_text_hot[i, validation_end_in_text[i]] = 1

assert val_label_hot[32, validation_start[32]] ==1 
assert val_label_hot[32, validation_end[32]] ==1 


tr_sentiment_seq = tokenizer.texts_to_sequences(train_sentiment)
tr_sentiment_seq = np.array(tr_sentiment_seq)
tr_sentiment_padded = pad_sequences(tr_sentiment_seq, maxlen=max_length-1, padding='post', truncating='post')

val_sentiment_seq = np.array(tokenizer.texts_to_sequences(val_sentiment))

# metric for QA
# the size of the intersection divided by the size of the union of the sample sets:
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def load_embedding(file_path, vocab, embedding_size):
  # vocab: {word: idx}
  print('loading embedding...')
  embedding_matrix = np.zeros((len(vocab)+1, embedding_size ))
  with open(file_path, encoding='utf8') as f:
    for line in f:
      # Each line will be a word and a list of floats, separated by spaces.
      # If the word is in your vocabulary, create a numpy array from the list of floats.
      # Assign the array to the correct row of embedding_matrix.
      values = line.split()
      word = values[0].lower()
      coefs = np.asarray(values[1:], dtype='float32')
      if word in vocab.keys():
        embedding_matrix[vocab[word]] = coefs
    embedding_matrix[vocab['<OOV>']] = np.random.rand(embedding_size)
    return embedding_matrix

# use the word_index to retrieve the transformed textual representation with OOV items
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_text(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
# print(decode_text(train_padded[0]))
# print('---')
# print(train_texts[0])
print('length of vocabulary: ', len(word_index))

embed_matrix = load_embedding(embedding_file_path, word_index, 25)
# assert embed_matrix.shape[0] == len(word_index)

# print('neutral ', embed_matrix[word_index['neutral']])
# print('positive ', embed_matrix[word_index['positive']])
# print('negative ', embed_matrix[word_index['negative']])

# ===========================================
# ============= MODELS ======================
# ===========================================
# an LSTM and learn word representations from the data

vocab_size= 29937
embedding_dim = 25

# baseline model takes pseudosentences as inputs
# model2 takes embedded text and auxiliary sentiment input
# model3 takes embedded text and embedded sentiment

# functional model layer defined below
tweet_input = Input(shape=(149,), name='tweet')
sentiment_input_oh = Input(shape=(3,), name='sentiment_input_one_hot')
# if using model3, need to specify batch for tweet input and adjust training/validation examples
tweet_input_batched = Input(batch_shape=(64,149,), name='tweet_batched')
senti_input = Input(batch_shape=(64,1,), name='sentiment_words_input')
comb_input = Input(shape=(max_length,))

comb_embed = Embedding(output_dim=embedding_dim, input_dim=vocab_size, mask_zero=True, 
                       weights=[embed_matrix], trainable=False, input_length=max_length)(comb_input)

tweet_embed = Embedding(output_dim=embedding_dim, 
                        input_dim=vocab_size, mask_zero=True,weights=[embed_matrix],
                        trainable=False, input_length=max_length-1)(tweet_input)  # (None, 149, 25)
tweet_embed_batched = Embedding(output_dim=embedding_dim,
                        input_dim=vocab_size, mask_zero=True,weights=[embed_matrix],
                        trainable=False, input_length=max_length-1)(tweet_input_batched)  # (64, 149, 25)

senti_embed = Embedding(output_dim=embedding_dim,  input_dim=vocab_size,
                        mask_zero=True,
                        weights=[embed_matrix],
                        trainable=True, input_length=1)(senti_input)

concat = concatenate([tweet_embed_batched, senti_embed], axis=1) # concatenate the sentiment 3 items on the 1st dim
x = LSTM(128, return_sequences=True)(comb_embed)
x2 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.02))(tweet_embed)
x3 = Bidirectional(LSTM(128, return_sequences=True))(concat)
x = LSTM(192, return_sequences=True)(x)
x2 = Bidirectional(LSTM(192, return_sequences=True))(x2)
x3 = Bidirectional(LSTM(192, return_sequences=True))(x3)
x = LSTM(256)(x) # (256,)
x2 = LSTM(256)(x2)
x3 = LSTM(256)(x3)
auxiliary = concatenate([x2, sentiment_input_oh])
#-------
start_logits_baseline = Dense(max_length, activation='softmax')(x)
after_start_baseline = concatenate([x, start_logits_baseline])
end_logits_baseline = Dense(max_length, activation='softmax')(after_start_baseline)
#------
start_logits2 = Dense(max_length-1, activation='softmax')(auxiliary)
after_start2 = concatenate([auxiliary, start_logits2])
end_logits2 = Dense(max_length-1, activation='softmax')(after_start2)
#------
start_logits3 = Dense(max_length-1, activation='softmax')(x3)
after_start3 = concatenate([x3, start_logits3])
end_logits3 = Dense(max_length-1, activation='softmax')(after_start3)


# baseline
model = Model(input=comb_input, output=[start_logits_baseline, end_logits_baseline])
model.compile(optimizer='adam', loss='categorical_crossentropy',
              loss_weights=[0.04, 0.04], metrics=['accuracy','cosine_proximity']
              )
print(model.summary())

##########################

model2 = Model(input=[tweet_input, sentiment_input_oh], output=[start_logits2, end_logits2])
# optimizer = Adam
model2.compile(optimizer='adam', loss='categorical_crossentropy',
               loss_weights=[0.04, 0.04], metrics=['accuracy','cosine_proximity'] )
print(model2.summary())
# keras.utils.plot_model(model2, 'lstm_aux_input.png', show_shapes=True)

##########################
model3 = Model(input=[tweet_input_batched, senti_input], output=[start_logits3, end_logits3])
# optimizer = Adam
model3.compile(optimizer='adam', loss='categorical_crossentropy', 
               loss_weights=[0.04, 0.04], metrics=['accuracy','cosine_proximity'] )
print(model3.summary())
# keras.utils.plot_model(model3, 'lstm_both_embed.png', show_shapes=True)

# history = model2.fit(train_padded, [train_start_hot, train_end_hot], epochs=10, verbose=2, batch_size=64)

print('training model2, with auxiliary sentiment input...\n')
history2 = model2.fit([train_padded_tweet, train_sentiment_oh], [train_start_in_text_hot, train_end_in_text_hot],
                     epochs=10, verbose=2, batch_size=64
                     )
#
# history3 = model3.fit([train_padded_tweet, tr_sentiment_seq],
#                      [train_start_in_text_hot, train_end_in_text_hot],
#                      epochs=10, verbose=2, batch_size=64
#                      )


###########
# example prediction on model2 on validation
# [:4096], val_sentiment_seq
predictions = model2.predict([validation_padded_tweet, val_sentiment_oh], batch_size=64)
print('pred[0] and [1] shapes: ', predictions[0].shape, predictions[1].shape)

# getting the predicted indices and turning into words
ans_start = np.zeros((predictions[0].shape[0],), dtype=np.int32)
ans_end = np.zeros((predictions[0].shape[0],),dtype=np.int32) 
for i in range(predictions[0].shape[0]):
    ans_start[i] = predictions[0][i, :].argmax()
    ans_end[i] = predictions[1][i, :].argmax()
print('ans_start shape: ', ans_start.shape)
print('ans_end shape: ',ans_end.shape)
print(ans_start.min(), ans_start.max(), ans_end.min(), ans_end.max())

all_scores = 0

print(val_tweets[:5])
pred_answers = []
gold_answers = []
for i, text in enumerate(val_tweets):
  # print(text)
  # print(ans_start[i], validation_start_in_text[i])
  # print(ans_end[i], validation_end_in_text[i])
  # turn indices into answer
  # if the sentiment is neutral than output the whole sentence
  if val_sentiment[i] == 'neutral':
    val_pred_ans = text
  else:
    val_pred_ans = text[ans_start[i]:ans_end[i]+1]
  pred_answers.append(val_pred_ans)
  val_gold_ans = text[validation_start_in_text[i]:validation_end_in_text[i]+1]
  gold_answers.append(val_gold_ans)
  j_score = jaccard(val_gold_ans, val_pred_ans)
  all_scores += j_score

avg_j_score = all_scores/len(val_tweets)
print('avg. jaccard score on validation: ', avg_j_score, '# validation: ', len(val_tweets))
print('example pred: ',pred_answers[15], '\nExample gold: ', gold_answers[15])

