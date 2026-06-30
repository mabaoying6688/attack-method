# coding: utf-8

import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import torch.nn.functional as F

class ImprovedAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super(ImprovedAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads       
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** 0.5
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()        
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)        
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)        
        attended = torch.matmul(attn_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        attended = self.out_proj(attended)        
        return attended


class LSTMEncoder(nn.Module):
    counter1 = 0
    def __init__(self, embedding_dim, hidden_dim, n_layers, drop_prob=0.3, brnn=True):
        super(LSTMEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = brnn
        LSTMEncoder.counter1 = 0
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 
                          n_layers, 
                          dropout=drop_prob, 
                          bidirectional=brnn,
                          batch_first=False)
        
    def forward(self, input, hidden=None):
        LSTMEncoder.counter1 += 1 
        # print("LSTMEncoder.counter1: ",LSTMEncoder.counter1)
        return self.lstm(input, hidden)


class LSTMClassifier(nn.Module):
    counter = 0
    def __init__(self, vocab_size, embedding_size, encoder, hidden_dim, num_classes, max_len, dropout_p=0.3, attn=True):
        super(LSTMClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding_dropout = nn.Dropout(0.1)
        self.embedding_norm = nn.LayerNorm(embedding_size)
        
        self.encoder = encoder
        self.hidden_dim = hidden_dim * 2 if self.encoder.bidirectional else hidden_dim
        self.max_len = max_len
        self.attn = attn
        LSTMClassifier.counter = 0

        if self.attn:
            self.attention = ImprovedAttention(self.hidden_dim, num_heads=8, dropout=0.1)
            self.attention_norm = nn.LayerNorm(self.hidden_dim)
  
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 512),  
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p * 0.5),
            nn.Linear(256, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout_p)
        
        size = 0
        for p in self.parameters():
            size += p.nelement()
        print('Total param size: {}'.format(size))
      
    def forward(self, inputs):
        LSTMClassifier.counter += 1   
        emb = self.embedding_norm(self.embedding_dropout(self.embedding(inputs)))
        
        outputs, hidden = self.encoder(emb)
        
        if self.attn:
            outputs_permuted = outputs.permute(1, 0, 2)  # (seq_len, batch, hidden) -> (batch, seq_len, hidden)
            attended = self.attention(outputs_permuted)
            attended = self.attention_norm(attended + outputs_permuted)  
            mean_pool = torch.mean(attended, dim=1)
            max_pool, _ = torch.max(attended, dim=1)
            features = torch.cat([mean_pool, max_pool], dim=1)
            
        else:
            features = torch.mean(outputs, dim=0)
            
        drop = self.dropout(features)
        logits = self.classifier(drop)
        return logits, emb

    def prob(self, inputs):
        logits = self.forward(inputs)[0]
        prob = nn.Softmax(dim=1)(logits)
        return prob
        
    def grad(self, inputs, labels, loss_fn):
        savep1 = self.encoder.lstm.dropout
        savep2 = self.dropout.p
        self.encoder.lstm.dropout = 0
        self.dropout.p = 0
        self.zero_grad()
        logits, emb = self.forward(inputs)
        emb.retain_grad()
        loss = loss_fn(logits, labels)
        loss.backward()
        self.encoder.lstm.dropout = savep1
        self.dropout.p = savep2
        return emb.grad.permute([1, 0, 2])


class GRUEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers, drop_prob=0.3, brnn=True):
        super(GRUEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = brnn
        
        self.gru = nn.GRU(embedding_dim, hidden_dim, 
                        n_layers, 
                        dropout=drop_prob, 
                        bidirectional=brnn,
                        batch_first=False)
        
    def forward(self, input, hidden=None):
        return self.gru(input, hidden)


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_size, encoder, hidden_dim, num_classes, max_len, dropout_p=0.3, attn=True):
        super(GRUClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding_dropout = nn.Dropout(0.1)
        self.embedding_norm = nn.LayerNorm(embedding_size)
        
        self.encoder = encoder
        self.hidden_dim = hidden_dim * 2 if self.encoder.bidirectional else hidden_dim
        self.max_len = max_len
        self.attn = attn
        
        if self.attn:
            self.attention = ImprovedAttention(self.hidden_dim, num_heads=8, dropout=0.1)
            self.attention_norm = nn.LayerNorm(self.hidden_dim)
            
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p * 0.5),
            nn.Linear(256, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout_p)
        
        size = 0
        for p in self.parameters():
            size += p.nelement()
        print('Total param size: {}'.format(size))
        
    def forward(self, inputs):
        emb = self.embedding_norm(self.embedding_dropout(self.embedding(inputs)))
        
        outputs, hidden = self.encoder(emb)
        
        if self.attn:
            outputs_permuted = outputs.permute(1, 0, 2)
            attended = self.attention(outputs_permuted)
            attended = self.attention_norm(attended + outputs_permuted)  # 残差连接
            mean_pool = torch.mean(attended, dim=1)
            max_pool, _ = torch.max(attended, dim=1)
            features = torch.cat([mean_pool, max_pool], dim=1)
            
        else:
            features = torch.mean(outputs, dim=0)
            
        drop = self.dropout(features)
        logits = self.classifier(drop)
        return logits, emb
    
    def prob(self, inputs):
        logits = self.forward(inputs)[0]
        prob = nn.Softmax(dim=1)(logits)
        return prob
        
    def grad(self, inputs, labels, loss_fn):
        savep1 = self.encoder.gru.dropout
        savep2 = self.dropout.p
        self.encoder.gru.dropout = 0
        self.dropout.p = 0
        self.zero_grad()
        logits, emb = self.forward(inputs)
        emb.retain_grad()
        loss = loss_fn(logits, labels)
        loss.backward()
        self.encoder.gru.dropout = savep1
        self.dropout.p = savep2
        return emb.grad.permute([1, 0, 2])