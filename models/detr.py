import torch
from torch import nn, Tensor
import torch.nn.functional as F
from transformers import LEDModel


class Transformer(nn.Module):

    def __init__(self, model):
        super().__init__()

        self.model = model
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder

    def forward(self, enc_input_ids, query_embed):

        enc = self.encoder(enc_input_ids)
        tgt = torch.zeros_like(query_embed)
        tgt = tgt.unsqueeze(-1).permute(2, 0, 1)
        tgt = tgt + query_embed
        dec = self.decoder(inputs_embeds=tgt, encoder_hidden_states=enc['last_hidden_state'])

        return dec


class DETR(nn.Module):

    def __init__(self, num_classes, num_queries, hidden_dim=768):
        super().__init__()

        self.transformer = Transformer()
        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = MLP(hidden_dim, hidden_dim, 2, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

    def forward(self, inputs_ids):
        h = self.transformer(inputs_ids, self.query_embed.weight)['last_hidden_state']

        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': self.linear_class(h),
                'pred_boxes': self.linear_bbox(h).sigmoid()}


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
