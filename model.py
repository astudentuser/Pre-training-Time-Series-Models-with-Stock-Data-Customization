import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(SelfAttentionLayer, self).__init__()
        assert hidden_size % num_heads == 0, "Hidden size must be divisible by the number of heads."
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.fc_q = nn.Linear(hidden_size, hidden_size)
        self.fc_k = nn.Linear(hidden_size, hidden_size)
        self.fc_v = nn.Linear(hidden_size, hidden_size)

        self.fc_o = nn.Linear(hidden_size, hidden_size)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = self.fc_q(query)  # (batch_size, seq_len, hidden_size)
        K = self.fc_k(key)  # (batch_size, seq_len, hidden_size)
        V = self.fc_v(value)  # (batch_size, seq_len, hidden_size)

        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / torch.sqrt(
            torch.tensor(self.head_dim).float())

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = F.softmax(energy, dim=-1)
        output = torch.matmul(attention, V)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1,
                                                              self.hidden_size)
        output = self.fc_o(output)

        return output


class PositionwiseFeedforward(nn.Module):
    def __init__(self, hidden_size, ff_hidden_size):
        super(PositionwiseFeedforward, self).__init__()
        self.fc1 = nn.Linear(hidden_size, ff_hidden_size)
        self.fc2 = nn.Linear(ff_hidden_size, hidden_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class FeatExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, num_feat_att_layers, num_heads, days, dropout=0.1):
        super(FeatExtractor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_feat_att_layers = num_feat_att_layers
        self.num_heads = num_heads
        self.days = days

        self.embedding = nn.Linear(input_size, hidden_size)
        self.positional_encoding = nn.Parameter(torch.randn(1, days, hidden_size))
        self.attention_layers = nn.ModuleList(
            [SelfAttentionLayer(hidden_size, num_heads) for _ in range(num_feat_att_layers)])
        self.feedforward_layers = nn.ModuleList(
            [PositionwiseFeedforward(hidden_size, hidden_size * 4) for _ in range(num_feat_att_layers)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded += self.positional_encoding[:, :x.size(1), :]

        for i in range(self.num_feat_att_layers):
            attention_output = self.attention_layers[i](embedded, embedded, embedded)
            embedded = embedded + self.dropout(attention_output)
            feedforward_output = self.feedforward_layers[i](embedded)
            embedded = embedded + self.dropout(feedforward_output)

        return embedded


class TransformerStockPrediction(nn.Module):
    def __init__(self, input_size, num_class, hidden_size, num_feat_att_layers, num_pre_att_layers, num_heads, days, dropout=0.1):
        super(TransformerStockPrediction, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_feat_att_layers = num_feat_att_layers
        self.num_pre_att_layers = num_pre_att_layers
        self.num_heads = num_heads
        self.days = days

        self.feature_extractor = FeatExtractor(input_size, hidden_size, num_feat_att_layers, num_heads, days, dropout)
        self.attention_layers = nn.ModuleList(
            [SelfAttentionLayer(hidden_size, num_heads) for _ in range(num_pre_att_layers)])
        self.feedforward_layers = nn.ModuleList(
            [PositionwiseFeedforward(hidden_size, hidden_size * 4) for _ in range(num_pre_att_layers)])
        self.fc = nn.Linear(hidden_size, num_class)
        # Pretrain output layer
        self.pretrain_task = ''
        self.pretrain_outlayers = nn.ModuleDict({})
        self.finetune = False

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        embedded = self.feature_extractor(x)

        for i in range(self.num_pre_att_layers):
            attention_output = self.attention_layers[i](embedded, embedded, embedded)
            embedded = embedded + self.dropout(attention_output)
            feedforward_output = self.feedforward_layers[i](embedded)
            embedded = embedded + self.dropout(feedforward_output)

        pooled = torch.mean(embedded, dim=1)

        if self.pretrain_task == '':
            output = self.fc(pooled)
        else:
            output = self.pretrain_outlayers[self.pretrain_task](pooled)

        return output

    def add_outlayer(self, name, num_class, device):
        if name not in self.pretrain_outlayers:
            self.pretrain_outlayers[name] = nn.Linear(self.hidden_size, num_class).to(device)
        else:
            print(f"Output layer '{name}' already exists.")

    def change_finetune_mode(self, mode, freezing='embedding'):
        self.finetune = mode
        if mode:
            if freezing == 'all':
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False

            elif freezing == 'embedding':
                for param in self.feature_extractor.embedding.parameters():
                    param.requires_grad = False

            elif freezing == 'embedding_attention':
                for param in self.feature_extractor.embedding.parameters():
                    param.requires_grad = False
                for param in self.feature_extractor.attention_layers.parameters():
                    param.requires_grad = False
        else:
            for param in self.feature_extractor.parameters():
                param.requires_grad = True

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path))
