from config import TRAIN_SET, BERT_BASE_TOKENIZER, SENT_DIS, CLEANED_TRAIN_SET, SENTIMENT_DIS, SENTIMENT_CONDITION_DIS
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

dataset = pd.read_csv(TRAIN_SET).dropna().reset_index(drop=True)
sentences = dataset.text.values
lens = [len(BERT_BASE_TOKENIZER.encode(sent).tokens) for sent in sentences]
print(f"Max : {max(lens)}.")
print(f"Avg : {sum(lens) / len(lens)}.")
drop_indices = [i for i, j in enumerate(lens) if j > 64]
cleaned_dataset = dataset.drop(drop_indices)
# cleaned_dataset.to_csv(CLEANED_TRAIN_SET, index=False)
lens_counter = Counter()
for i in lens:
    if i < 4:
        lens_counter['0-3'] += 1
    elif i >= 4 and i < 8:
        lens_counter['4-7'] += 1
    elif i >= 8 and i < 16:
        lens_counter['8-15'] += 1
    elif i >= 16 and i < 32:
        lens_counter['16-31'] += 1
    elif i >= 32 and i < 64:
        lens_counter['32-63'] += 1
    elif i >= 64 and i <= 128:
        lens_counter['64-128'] += 1

sorted_items = sorted(list(lens_counter.items()), key=lambda x:int(x[0].split('-')[1]))
keys = []
nums = []
for (key, num) in sorted_items:
    keys.append(key)
    nums.append(num)

fig, ax = plot.subplots(1, 1, figsize=(12, 8))
x_axis = np.arange(len(keys))
ax.bar(x_axis, nums, width=0.8)
ax.set_ylabel('Amount')
ax.set_xlabel('Sentence length')
ax.set_title('Sentences length distribution')
ax.set_xticks(x_axis)
ax.set_xticklabels(keys)
fig.savefig(SENT_DIS, format='svg')

neutral_samples = cleaned_dataset.query("sentiment == 'neutral'")
neutral_counter = Counter()
for i, j in zip(neutral_samples.text.values.tolist(), neutral_samples.selected_text.values.tolist()):
    if i.strip() == j.strip():
        neutral_counter['text == selected_text'] += 1
    else:
        neutral_counter['text != selected_text'] += 1
neutral_keys = []
neutral_nums = []
for (key, num) in sorted(list(neutral_counter.items())):
    neutral_keys.append(key)
    neutral_nums.append(num)
neutral_nums = [i / sum(neutral_nums) for i in neutral_nums]

positive_samples = cleaned_dataset.query("sentiment == 'positive'")
positive_counter = Counter()
for i, j in zip(positive_samples.text.values.tolist(), positive_samples.selected_text.values.tolist()):
    if i.strip() == j.strip():
        positive_counter['text == selected_text'] += 1
    else:
        positive_counter['text != selected_text'] += 1
positive_keys = []
positive_nums = []
for (key, num) in sorted(list(positive_counter.items())):
    positive_keys.append(key)
    positive_nums.append(num)
positive_nums = [i / sum(positive_nums) for i in positive_nums]

negative_samples = cleaned_dataset.query("sentiment == 'negative'")
negative_counter = Counter()
for i, j in zip(negative_samples.text.values.tolist(), negative_samples.selected_text.values.tolist()):
    if i.strip() == j.strip():
        negative_counter['text == selected_text'] += 1
    else:
        negative_counter['text != selected_text'] += 1
negative_keys = []
negative_nums = []
for (key, num) in sorted(list(negative_counter.items())):
    negative_keys.append(key)
    negative_nums.append(num)
negative_nums = [i / sum(negative_nums) for i in negative_nums]

fig, ax = plot.subplots(1, 3, figsize=(18, 4), squeeze=False)
ax[0][0].set_title('Neutral sample distribution')
ax[0][1].set_title('Positive sample distribution')
ax[0][2].set_title('Negative sample distribution')

ax[0][0].set_xlabel('Equal or not')
ax[0][1].set_xlabel('Equal or not')
ax[0][2].set_xlabel('Equal or not')

ax[0][0].set_ylabel('Amount')
ax[0][1].set_ylabel('Amount')
ax[0][2].set_ylabel('Amount')

ax[0][0].bar(neutral_keys, neutral_nums, width=0.5)
ax[0][1].bar(positive_keys, positive_nums, width=0.5)
ax[0][2].bar(negative_keys, negative_nums, width=0.5)

fig.savefig(SENTIMENT_DIS, format='svg')

split_lens = [len(sent.strip().split()) for sent in sentences]
larger_1_indices = [i for i, j in enumerate(split_lens) if j > 1]
less_2_dataset = dataset.drop(larger_1_indices)
less_2_sentiment = less_2_dataset.sentiment.values.tolist()
less_2_dic = Counter()
for sentiment in less_2_sentiment:
    less_2_dic[sentiment] += 1
less_2_keys = []
less_2_nums = []
for (key, num) in sorted(list(less_2_dic.items())):
    less_2_keys.append(key)
    less_2_nums.append(num)
less_2_nums = [i / sum(less_2_nums) for i in less_2_nums]

larger_2_indices = [i for i, j in enumerate(split_lens) if j > 2]
less_3_dataset = dataset.drop(larger_2_indices)
less_3_sentiment = less_3_dataset.sentiment.values.tolist()
less_3_dic = Counter()
for sentiment in less_3_sentiment:
    less_3_dic[sentiment] += 1
less_3_keys = []
less_3_nums = []
for (key, num) in sorted(list(less_3_dic.items())):
    less_3_keys.append(key)
    less_3_nums.append(num)
less_3_nums = [i / sum(less_3_nums) for i in less_3_nums]

larger_3_indices = [i for i, j in enumerate(split_lens) if j > 3]
less_4_dataset = dataset.drop(larger_3_indices)
less_4_sentiment = less_4_dataset.sentiment.values.tolist()
less_4_dic = Counter()
for sentiment in less_4_sentiment:
    less_4_dic[sentiment] += 1
less_4_keys = []
less_4_nums = []
for (key, num) in sorted(list(less_4_dic.items())):
    less_4_keys.append(key)
    less_4_nums.append(num)
less_4_nums = [i / sum(less_4_nums) for i in less_4_nums]

fig, ax = plot.subplots(1, 3, figsize=(18, 4), squeeze=False)
ax[0][0].set_title('less 2 sentiment distribution')
ax[0][1].set_title('less 3 sentiment distribution')
ax[0][2].set_title('less 4 sentiment distribution')

ax[0][0].set_xlabel('sentiment')
ax[0][1].set_xlabel('sentiment')
ax[0][2].set_xlabel('sentiment')

ax[0][0].set_ylabel('Amount')
ax[0][1].set_ylabel('Amount')
ax[0][2].set_ylabel('Amount')

ax[0][0].bar(less_2_keys, less_2_nums, width=0.5)
ax[0][1].bar(less_3_keys, less_3_nums, width=0.5)
ax[0][2].bar(less_4_keys, less_4_nums, width=0.5)

fig.savefig(SENTIMENT_CONDITION_DIS, format='svg')
