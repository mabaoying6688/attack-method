# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.init as init

class LSCNNClassifier(torch.nn.Module):
    counter=0
    def __init__(self, n_class, vocab_size, embed_width, n_conv, conv_size,
                lstm_size, n_lstm, bilstm=True, device=None):
        super(LSCNNClassifier, self).__init__()
        self.embed_width = self.x_size = embed_width
        self.device = device
        self.n_conv = n_conv
        self.lstm_size = lstm_size * 2 if bilstm else lstm_size
        self.vocab_size = vocab_size
    
        self.embed = nn.Embedding(vocab_size, embed_width, padding_idx=0)
        self.embed_dropout = nn.Dropout(0.1)
        self.embed_norm = nn.LayerNorm(embed_width)
        LSCNNClassifier.counter = 0
       
        self.conv = nn.Sequential(
            nn.Conv1d(embed_width, n_conv, conv_size, padding=(conv_size-1)//2),
            nn.BatchNorm1d(n_conv),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.conv_norm = nn.LayerNorm(n_conv)

        # LSTM
        self.lstm = nn.LSTM(
            n_conv, 
            lstm_size, 
            n_lstm, 
            bidirectional=bilstm,
            dropout=0.3 if n_lstm > 1 else 0
        )

    
        self.classify = nn.Sequential(
            nn.Linear(self.lstm_size, self.lstm_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.lstm_size // 2, n_class)
        )
        
      
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if param.dim() < 2:
                continue
            if 'weight' in name:
                if 'lstm' in name:
                    if 'weight_hh' in name:
                        init.orthogonal_(param)
                        param.data.mul_(0.01)
                    else:
                        init.xavier_uniform_(param, gain=0.1)
                elif 'embed' in name:
                    init.uniform_(param, -0.05, 0.05)
                else:
                    init.xavier_uniform_(param, gain=1.0)
            if 'bias' in name:
                init.constant_(param, 0.0)
        init.xavier_uniform_(self.classify[-1].weight, gain=0.1)
        init.constant_(self.classify[-1].bias, 0.0)

    def forward(self, x, l):
        bs, stmt_cnt, stmt_len = x.shape
        LSCNNClassifier.counter+=1

        if l.dim() == 1:
            if l.size(0) == stmt_cnt:
              
                l = l.unsqueeze(0).expand(bs, stmt_cnt).contiguous()
            else:
                raise ValueError(f"l shape {l.shape} not compatible with x shape {x.shape}")
        elif l.shape != (bs, stmt_cnt):
            raise ValueError(f"Expected l shape (bs, stmt_cnt), got {l.shape}")

        # Embedding
        embedding = self.embed(x)
        embedding = self.embed_dropout(embedding)
        embedding = self.embed_norm(embedding)
        embedding = embedding.reshape(bs * stmt_cnt, stmt_len, self.embed_width).transpose(1, 2)
        
        # Convolution + LayerNorm
        convolution = self.conv(embedding)
        convolution = self.conv_norm(convolution.transpose(1,2)).transpose(1,2)

        _l = l.reshape(bs * stmt_cnt)
        conv_mask = (torch.arange(stmt_len, device=self.device)[None, :] < _l[:, None])
        conv_mask = conv_mask.unsqueeze(1).expand(-1, self.n_conv, -1)
        convolution = convolution.masked_fill(~conv_mask, -1e4)
        
        pooling = nn.functional.adaptive_max_pool1d(convolution, 1)
        pooling = pooling.squeeze(-1).reshape(bs, stmt_cnt, self.n_conv)
        pooling = pooling.transpose(0, 1)

        h, _ = self.lstm(pooling)
        h = h.transpose(0, 1)
        h_mask = (l == -1).unsqueeze(-1).expand(-1, -1, self.lstm_size)
        h = h.masked_fill(h_mask, -1e4)

        h = h.transpose(1, 2)
        pooling2 = nn.functional.adaptive_max_pool1d(h, 1).squeeze(-1)
        logits = self.classify(pooling2)
        
        return logits

    def prob(self, x, l):
        with torch.no_grad():
            logits = self.forward(x, l)
            return nn.functional.softmax(logits, dim=-1)

    def grad(self, x, l, labels, loss_fn):
        LSCNNClassifier.counter+=1
        self.zero_grad()
        logits = self.forward(x, l)
        loss = loss_fn(logits, labels)
        loss.backward()
        return self.embed.weight.grad
