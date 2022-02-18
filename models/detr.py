from lib2to3.pgen2 import token
import torch
from torch import nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model

        self.encoder = self.model.encoder
        self.decoder = self.model.decoder

    def forward(self, enc_input_ids, query_embed, glob_enc_attn, glob_dec_attn):
        # Non mi è chiaro come dobbiamo assegnare le mask.
        # per la maschera dell'encoder consigliano di metter la global attention solo sul token
        # che definisce l'inizio <s>, mentre per il decoder non capisco.
        # Ditemi voi se capite meglio.
        # glob_enc_mask = torch.zeros(tokens.size()[1])
        # glob_enc_mask[0] = 1
        enc = self.encoder(enc_input_ids, global_attention_mask=glob_enc_attn)
        tgt = torch.zeros_like(query_embed)
        tgt = tgt.unsqueeze(-1).permute(2, 0, 1)
        tgt = tgt + query_embed
        dec = self.decoder(
            inputs_embeds=tgt,
            encoder_hidden_states=enc["last_hidden_state"],
            global_attention_mask=glob_dec_attn,
        )

        return dec


class DETR(nn.Module):
    def __init__(
        self, model, num_classes, num_queries, hidden_dim, transformer_hidden_dim=768
    ):
        super().__init__()

        self.transformer = Transformer(model)
        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(transformer_hidden_dim, num_classes + 1)
        self.linear_bbox = MLP(transformer_hidden_dim, hidden_dim, 2, 3)
        self.query_embed = nn.Embedding(num_queries, transformer_hidden_dim)
        # output positional encodings (object queries)
        self.query_pos = nn.parameter.Parameter(torch.rand(100, transformer_hidden_dim))
        self.num_queries = num_queries

    def forward(self, inputs_ids, glob_enc_attn, glob_dec_attn):
        h = self.transformer(
            inputs_ids, self.query_embed.weight, glob_enc_attn, glob_dec_attn
        )["last_hidden_state"]

        # finally project transformer outputs to class labels and bounding boxes
        return {
            "pred_logits": self.linear_class(h),
            "pred_boxes": self.linear_bbox(h).sigmoid(),
        }

    def set_transformer_trainable(self, trainable: bool):
        for param in self.transformer.parameters():
            param.requires_grad = trainable

    def transformer_parameters(self):
        return (p for n, p in self.named_parameters() if "transformer" in n and p.requires_grad)
    
    def last_layers_parameters(self):
        return (p for n, p in self.named_parameters() if "transformer" not in n and p.requires_grad)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PrepareInputs:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, docs):
        # tokens = [
        #     self.tokenizer(text, return_tensors="pt").input_ids.squeeze()
        #     for text in docs
        # ]
        # return torch.nn.utils.rnn.pad_sequence(
        #     tokens, batch_first=True, padding_value=0.0
        # )

        return self.tokenizer(docs, padding=True, return_tensors='pt').input_ids
