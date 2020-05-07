import config
import torch
import os
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from dataset import BertBaseTweetDataset
from models import BertBaseForQAWithSentiment
from functions import TrainWithSentiment, EvaluateWithSentiment, log_tracker

######## Hyperparameter ########
batch_size = 32
epochs = 20
lr = 2e-5
eps=1e-8
warmup_steps = 0
model_name = config.BERTBASEWITHSENTIMENT
split_ratio = 0.10
random_seed = 233
######## Hyperparameter ########

dataset = pd.read_csv(config.CLEANED_TRAIN_SET).dropna().reset_index(drop=True)
train_set, dev_set = train_test_split(
    dataset,
    test_size=split_ratio,
    random_state=random_seed,
    stratify=dataset.sentiment.values
)
train_dataset = BertBaseTweetDataset(
    text=train_set.text.values,
    selected_text=train_set.selected_text.values,
    sentiment=train_set.sentiment.values
)
dev_dataset = BertBaseTweetDataset(
    text=dev_set.text.values,
    selected_text=dev_set.selected_text.values,
    sentiment=dev_set.sentiment.values
)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size
)
dev_loader = DataLoader(
    dev_dataset,
    batch_size=batch_size
)
device = torch.device('cuda')
model = BertBaseForQAWithSentiment()
model.to(device)
train_steps = int(len(train_loader) * epochs)
optimizer = AdamW(model.parameters(), lr=lr, eps=eps)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=train_steps
)
best_jaccard = 0
log_address = os.path.join(config.LOG_FOLDER, model_name + ".csv")
jaccard_tracker = log_tracker(log_address)
for epoch in range(epochs):
    TrainWithSentiment(
        data_loader=train_loader,
        model=model,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler
    )
    jaccard = EvaluateWithSentiment(dev_loader, model, device, selection_method="maximum")
    print(f"Jaccard while {epoch + 1} out of {epochs} epoch : {jaccard}")
    jaccard_tracker.update(jaccard)
    if jaccard > best_jaccard:
        jaccard = str(jaccard)[2:5]
        checkpoint_name = model_name + "_" + jaccard + ".bin"
        save_path = os.path.join(config.CHECKPOINT_FOLDER, checkpoint_name)
        torch.save(model.state_dict(), save_path)
jaccard_tracker.store()