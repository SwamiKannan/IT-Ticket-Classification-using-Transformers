import transformers
from transformers import BertModel
import torch.nn as nn
import torch.nn.functional as F

class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        #self.bert.embeddings.word_embeddings.num_embeddings=40000
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(768, 256)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(256, 44)

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output1 = self.dropout1(pooled_output)
        linear_output1 = self.linear1(dropout_output1)
        dropout_output2 = self.dropout2(linear_output1)
        linear_output2 = self.linear2(dropout_output2)
        final_layer = F.softmax(linear_output2,dim=-1)
        return final_layer