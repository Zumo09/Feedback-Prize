from torch import nn
from transformers import LEDModel, LEDTokenizerFast


class DETR(nn.Module):

    def __init__(self, hidden_dim, num_classes, num_queries):

        super().__init__()

        self.tokenizer = LEDTokenizerFast.from_pretrained('allenai/led-base-16384')

        led = LEDModel.from_pretrained('allenai/led-base-16384')
        self.encoder = led.encoder
        self.decoder = led.decoder

        self.num_queries = num_queries
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

    def forward(self, text):
        ids = self.tokenizer(text, return_tensor='pt').input_ids
        hs = self.encoder(input_ids=ids)['last_hidden_state']
        hs = self.decoder(input_embeds=self.query_embeds.weigth, encoder_hidden_state=hs)['last_hidden_state']
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        # if self.aux_loss:
        #     out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord) # will we use auxiliary losses??
        return out


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.ReLU(layer(x)) if i < self.num_layers - 1 else layer(x)