import config
import torch
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

class BertBaseTweetDataset:
    def __init__(self, text, selected_text, sentiment):
        self.text = text
        self.selected_text = selected_text
        self.sentiment = sentiment
        self.max_len = config.MAX_LEN
        self.tokenizer = config.BERT_BASE_TOKENIZER

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = " ".join(self.text[index].split())
        selected_text = " ".join(self.selected_text[index].split())
        # Find character-level indices
        len_selected_text = len(selected_text)
        start_index = -1                                                            # Start of the selected text
        end_index = -1                                                              # End of the selected text
        for ind in [i for i, char in enumerate(text) if char == selected_text[0]]:
            if text[ind : ind + len_selected_text] == selected_text:
                start_index = ind
                end_index = ind + len_selected_text - 1                             # The index of the last character of the selected text
                break                                                               # The first match is the right match

        char_targets = [0] * len(text)                                              # Initialization for the character-level target
        if start_index != -1 and end_index != -1:
            for i in range(start_index, end_index + 1):
                if text[i] != " ":
                    char_targets[i] = 1

        encoded_text = self.tokenizer.encode(text)                                  # Token-level
        encoded_text_tokens = encoded_text.tokens
        encoded_text_ids = encoded_text.ids
        encoded_text_offsets = encoded_text.offsets[1:-1]                           # Get rid of the [CLS] and [SEP]

        targets = [0] * (len(encoded_text_tokens) - 2)                              # Initialization for the Token-level target
        for i, (start, end) in enumerate(encoded_text_offsets):
            if sum(char_targets[start: end]) > 0:
                targets[i] = 1
        targets = [0] + targets + [0]                                               # Add the mask for [CLS] and [SEP]
        targets_start = [0] * len(targets)
        targets_end = [0] * len(targets)
        nonzero_indices = np.nonzero(targets)[0]
        if len(nonzero_indices) != 0:
            targets_start[nonzero_indices[0]] = 1
            targets_end[nonzero_indices[-1]] = 1
        
        pad_len = self.max_len - len(encoded_text_ids)
        ids = encoded_text_ids + pad_len * [0]
        mask = [int(token_id) > 0 for token_id in ids]
        targets = targets + pad_len * [0]
        targets_start = targets_start + pad_len * [0]
        targets_end = targets_end + pad_len * [0]

        if self.sentiment[index] == "positive":
            sentiment = [1, 0, 0]
        elif self.sentiment[index] == "neutral":
            sentiment = [0, 1, 0]
        else:
            sentiment = [0, 0, 1]
        
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
            "targets_start": torch.tensor(targets_start, dtype=torch.long),
            "targets_end": torch.tensor(targets_end, dtype=torch.long),
            "text_tokens": " ".join(encoded_text_tokens),
            "sentiment": torch.tensor(sentiment, dtype=torch.long),
            "padding_len": torch.tensor(pad_len, dtype=torch.long),
            "original_text": self.text[index],
            "original_selected_text": self.selected_text[index],
            "original_sentiment": self.sentiment[index]
        }

if __name__ == "__main__":
    # dataset = pd.read_csv(config.TRAIN_SET).dropna().reset_index(drop=True)
    passs
