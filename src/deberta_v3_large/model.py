import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
#4) MSD with lat layer
class CustomModel4(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        self.dropout = nn.Dropout(cfg.dropout)
        self.high_dropout = nn.Dropout(cfg.high_dropout)

        self.dropouts = nn.ModuleList([
                            nn.Dropout(0.5) for _ in range(5)
            ])

        self.layer_norm = nn.LayerNorm(self.config.hidden_size)
        n_weights = self.config.num_hidden_layers + 1
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)
        self.fc = nn.Linear(self.config.hidden_size, 1)
        self._init_weights(self.fc)



    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs):
        outputs = self.model(**inputs)

        # hidden_layers = outputs[1]
        sequence_output = outputs[0]
        sequence_output = self.layer_norm(sequence_output)

        for i , dropout in enumerate(self.dropouts):
            if i==0:
                logits = self.fc(dropout(sequence_output))
            else :
                logits += self.fc(dropout(sequence_output))

        logits /= len(self.dropouts)

        return logits


#3 ) default linear model
class CustomModel3(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel(self.config)
        self.fc_dropout = nn.Dropout(cfg.fc_dropout)
        self.fc = nn.Linear(self.config.hidden_size, 1)
        self._init_weights(self.fc)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        return last_hidden_states

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(self.fc_dropout(feature))
        return output


class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel(self.config)
        self.fc_dropout = nn.Dropout(cfg.fc_dropout)
        self.fc = nn.Linear(self.config.hidden_size * 2 , 1)
        self._init_weights(self.fc)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def feature(self, inputs):
        outputs = self.model(**inputs)
        LAST_HIDDEN_LAYERS = 8
        outputs = outputs.hidden_states
        outputs = torch.stack(tuple(outputs[-i - 1] for i in range(LAST_HIDDEN_LAYERS)) , dim=0)
        out_mean = torch.mean(outputs , dim=0)
        out_max , _ = torch.max(outputs , dim =0)
        out = torch.cat((out_mean , out_max) , dim=-1)
        return out

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = torch.mean(torch.stack([
            self.fc(self.fc_dropout(feature))
            for _ in range(5)
            ], dim=0), dim=0)
        return output

