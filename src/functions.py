import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from config import BERT_BASE_TOKENIZER
from collections import Counter
from tqdm import tqdm

def LossWithOutSentiment(predicted_start, predicted_end, true_start, true_end):
    loss_start = nn.BCEWithLogitsLoss()(predicted_start, true_start)
    loss_end = nn.BCEWithLogitsLoss()(predicted_end, true_end)
    return (loss_start + loss_end) / 2

def LossWithSentiment(predicted_sentiment, predicted_start, predicted_end, true_sentiment, true_start, true_end, alpha=0.2):
    loss_sentiment = nn.BCEWithLogitsLoss()(predicted_sentiment.view(-1, 3), true_sentiment)
    loss_start = nn.BCEWithLogitsLoss()(predicted_start, true_start)
    loss_end = nn.BCEWithLogitsLoss()(predicted_end, true_end)
    return (1 - alpha) * (loss_sentiment + loss_start) + alpha * loss_end

def TrainWithSentiment(data_loader, model, optimizer, device, scheduler):
    model.train()
    loss_tracker = AverageMeter()
    t = tqdm(data_loader, total=len(data_loader))
    for _, sample in enumerate(t):
        ids = sample["ids"]
        mask = sample["mask"]
        targets_start = sample["targets_start"]
        targets_end = sample["targets_end"]
        sentiment = sample["sentiment"]
        
        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.float)
        targets_end = targets_end.to(device, dtype=torch.float)
        sentiment = sentiment.to(device, dtype=torch.float)
        
        optimizer.zero_grad()
        predicted_sentiment, predicted_start, predicted_end = model(
            ids=ids,
            mask=mask,

        )
        loss = LossWithSentiment(
            predicted_sentiment=predicted_sentiment,
            predicted_start=predicted_start,
            predicted_end=predicted_end,
            true_sentiment=sentiment,
            true_start=targets_start,
            true_end=targets_end
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_tracker.update(loss.item(), ids.size(0))
        t.set_postfix(loss=loss_tracker.avg)

def TrainWithOutSentiment(data_loader, model, optimizer, device, scheduler):
    model.train()
    loss_tracker = AverageMeter()
    t = tqdm(data_loader, total=len(data_loader))
    for _, sample in enumerate(t):
        ids = sample["ids"]
        mask = sample["mask"]
        targets_start = sample["targets_start"]
        targets_end = sample["targets_end"]
        
        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.float)
        targets_end = targets_end.to(device, dtype=torch.float)
        
        optimizer.zero_grad()
        predicted_start, predicted_end = model(
            ids=ids,
            mask=mask
        )
        loss = LossWithOutSentiment(
            predicted_start=predicted_start,
            predicted_end=predicted_end,
            true_start=targets_start,
            true_end=targets_end
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_tracker.update(loss.item(), ids.size(0))
        t.set_postfix(loss=loss_tracker.avg)

def EvaluateWithSentiment(data_loader, model, device, selection_method="candidate"):
    assert selection_method in ["candidate", "maximum"], f"{selection_method} not found."
    model.eval()
    predict_starts = []
    predict_ends = []
    ids_list = []
    padding_lens = []
    selected_list = []
    original_sentiments = []
    original_texts = []
    for _, sample in enumerate(data_loader):
        ids = sample["ids"]
        mask = sample["mask"]
        padding_len = sample["padding_len"]
        original_selected = sample["original_selected_text"]
        sentiment = sample["original_sentiment"]
        original_text = sample["original_text"]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)

        _, predicted_start, predicted_end = model(
            ids=ids,
            mask=mask
        )
        predict_starts.append(torch.softmax(predicted_start, dim=1).cpu().detach().numpy())
        predict_ends.append(torch.softmax(predicted_end, dim=1).cpu().detach().numpy())
        ids_list.append(ids.cpu().detach().numpy().tolist())
        padding_lens.extend(padding_len.cpu().detach().numpy().tolist())
        selected_list.extend(original_selected)
        original_sentiments.extend(sentiment)
        original_texts.extend(original_text)
    predict_starts = np.vstack(predict_starts)
    predict_ends = np.vstack(predict_ends)
    ids_list = np.vstack(ids_list)
    jaccards = calculate_jaccards(
        selected_list,
        padding_lens,
        original_sentiments,
        original_texts,
        predict_starts,
        predict_ends,
        ids_list,
        selection_method,
        threshold=0.3
    )
    mean_jac = np.mean(jaccards, dtype=np.float32)
    return mean_jac

def EvaluateWithOutSentiment(data_loader, model, device, selection_method="candidate"):
    assert selection_method in ["candidate", "maximum"], f"{selection_method} not found."
    model.eval()
    predict_starts = []
    predict_ends = []
    ids_list = []
    padding_lens = []
    selected_list = []
    original_sentiments = []
    original_texts = []
    for _, sample in enumerate(data_loader):
        ids = sample["ids"]
        mask = sample["mask"]
        padding_len = sample["padding_len"]
        original_selected = sample["original_selected_text"]
        sentiment = sample["original_sentiment"]
        original_text = sample["original_text"]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)

        predicted_start, predicted_end = model(
            ids=ids,
            mask=mask
        )
        predict_starts.append(torch.softmax(predicted_start, dim=1).cpu().detach().numpy())
        predict_ends.append(torch.softmax(predicted_end, dim=1).cpu().detach().numpy())
        ids_list.append(ids.cpu().detach().numpy().tolist())
        padding_lens.extend(padding_len.cpu().detach().numpy().tolist())
        selected_list.extend(original_selected)
        original_sentiments.extend(sentiment)
        original_texts.extend(original_text)
    predict_starts = np.vstack(predict_starts)
    predict_ends = np.vstack(predict_ends)
    ids_list = np.vstack(ids_list)
    jaccards = calculate_jaccards(
        selected_list,
        padding_lens,
        original_sentiments,
        original_texts,
        predict_starts,
        predict_ends,
        ids_list,
        selection_method,
        threshold=0.3
    )
    mean_jac = np.mean(jaccards, dtype=np.float32)
    return mean_jac


def calculate_jaccards(
    selected_list,
    padding_lens,
    original_sentiments,
    original_texts,
    predict_starts,
    predict_ends,
    ids_list,
    selection_method,
    threshold=None
):
    jaccards = []
    if selection_method == "candidate":
        assert threshold != None, "threhold shouldn't be None while using candidate method."
        for i in range(len(ids_list)):
            selected_string = selected_list[i]
            padding_len = padding_lens[i]
            sentiment_value = original_sentiments[i]
            original_text_value = original_texts[i]
            if padding_len > 0:
                mask_starts = predict_starts[i, :][:-padding_len] >= threshold           # Mask the result candidates
                mask_ends = predict_ends[i, :][:-padding_len] >= threshold
                ids = ids_list[i, :][:-padding_len]
            else:
                mask_starts = predict_starts[i, :] >= threshold
                mask_ends = predict_ends[i, :] >= threshold
                ids = ids_list[i, :]
            mask = [0] * len(mask_starts)                                                # len(mask_start) == len(mask_end)
            index_starts = np.nonzero(mask_starts)[0]
            index_ends = np.nonzero(mask_ends)[0]
            if len(index_starts) > 0:
                index_start = index_starts[0]
                if len(index_ends) > 0:
                    index_end = index_ends[0]
                else:
                    index_end = index_start
            else:
                index_start = 0
                index_end = 0
            for i in range(index_start, index_end + 1):
                mask[i] = 1
            output_ids = [j for i, j in enumerate(ids) if mask[i] == 1]
            output = BERT_BASE_TOKENIZER.decode(output_ids)
            output = output.strip()
            if sentiment_value == "neutral" or len(original_text_value.split()) < 2:
                output = original_text_value
            jac = jaccard(selected_string, output)
            jaccards.append(jac)
    elif selection_method == "maximum":
        for i in range(len(ids_list)):
            selected_string = selected_list[i]
            padding_len = padding_lens[i]
            sentiment_value = original_sentiments[i]
            original_text_value = original_texts[i]
            if padding_len > 0:
                mask_length = len(predict_starts[i, :][:-padding_len])
                ids = ids_list[i, :][:-padding_len]
            else:
                mask_length = len(predict_starts[i, :])
                ids = ids_list[i, :]
            mask = [0] * mask_length
            index_start = np.argmax(predict_starts[i, :])
            index_end = np.argmax(predict_ends[i, :])
            if index_start > index_end:
                index_start = 0
            for i in range(index_start, index_end + 1):
                mask[i] = 1
            output_ids = [j for i, j in enumerate(ids) if mask[i] == 1]
            output = BERT_BASE_TOKENIZER.decode(output_ids)
            output = output.strip()
            if sentiment_value == "neutral" or len(original_text_value.split()) < 2:
                output = original_text_value
            jac = jaccard(selected_string, output)
            jaccards.append(jac)
    return jaccards

def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class log_tracker:
    def __init__(self, address):
        self.reset(address)
    
    def reset(self, address):
        self.vals = []
        self.address = address
    
    def update(self, val):
        self.vals.append(val)
    
    def store(self):
        self.vals = [str(val) + "\n" for val in self.vals]
        with open(self.address, 'w') as writer:
            writer.writelines(self.vals)