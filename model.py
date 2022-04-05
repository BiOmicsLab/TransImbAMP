import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tape import ProteinBertAbstractModel, ProteinBertModel, ProteinConfig
from tape.models.modeling_utils import SimpleMLP
from torch.nn.utils.weight_norm import weight_norm


# class AMPLevel1Head(nn.Module):
#     def __init__(self, hidden_size: int, num_labels: int):
#         super().__init__()
#         self.classify = SimpleMLP(hidden_size, 512, num_labels)

#     def forward(self, pooled_output, targets=None):
#         logits = self.classify(pooled_output)
#         outputs = (logits, )

#         if targets is not None:
#             outputs = logits, targets

#         return outputs  # logits, (targets)


class BERTAMP(nn.Module):

    def __init__(self, 
                 linsize:int = 512, 
                 lindropout: float = 0.8,
                 num_labels: int = 2, # 2 under simple binary classification
                 pretrained: bool = True,
                 bert_config = None,
                 bert_frozen = True):
        super().__init__()

        if pretrained:
            model_path = "./pretrained_model/tape-bert-base.pkl"
            if os.path.exists(model_path):
                self.bert = torch.load(model_path)
            else:
                self.bert = ProteinBertModel.from_pretrained('bert-base')
        else:
            self.bert = ProteinBertModel(bert_config)

        hidden_size = self.bert.config.hidden_size

        self.classify = nn.Sequential(
            weight_norm(nn.Linear(hidden_size, linsize), dim=None),
            nn.LeakyReLU(),
            nn.Dropout(lindropout, inplace=True),
            weight_norm(nn.Linear(linsize, num_labels), dim=None))
        # Frozen the former layer
        if bert_frozen:
            for p in self.bert.parameters():
                p.requires_grad = False     

        # initialize weight for the MLP classification
        for p in self.classify.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, input_ids, input_mask=None):

        outputs = self.bert(input_ids, input_mask=input_mask)

        sequence_output, _ = outputs[:2]

        average = torch.mean(sequence_output, dim=1)
        outputs = (self.classify(average), average) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs
