import config
import transformers
import torch.nn as nn

class BertBaseForQAWithoutSentiment(nn.Module):
    def __init__(self):
        super(BertBaseForQAWithoutSentiment, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_BASE)
        self.linear4span = nn.Linear(768, 2)

    def forward(self, ids, mask):
        seq_output, pooled_output = self.bert(
            input_ids=ids,
            attention_mask=mask
        )                                                                           # seq_output: (batch_size, seq_len, 768)
        logits = self.linear4span(seq_output)                                       # logits: (batch_size, seq_len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)                          # Split into two equal size part base on last dimension
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits

class BertBaseForQAWithSentiment(nn.Module):
    def __init__(self):
        super(BertBaseForQAWithSentiment, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_BASE)
        self.linear4span = nn.Linear(768, 2)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 3)
    
    def forward(self, ids, mask):
        seq_output, pooled_output = self.bert(
            input_ids=ids,
            attention_mask=mask
        )
        pooled_output = self.dropout(pooled_output)
        sentiment_logits = self.classifier(pooled_output)
        logits = self.linear4span(seq_output)                                       # logits: (batch_size, seq_len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)                          # Split into two equal size part base on last dimension
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return sentiment_logits, start_logits, end_logits