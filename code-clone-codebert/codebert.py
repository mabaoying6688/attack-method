# -*- coding: utf-8 -*-


from __future__ import absolute_import, division, print_function
import torch.nn.functional as F
import argparse
import glob
import logging
from dataset import OJ104
from torch import optim
import torch
from torch import nn
from transformers import RobertaForSequenceClassification, RobertaTokenizer

logger = logging.getLogger(__name__)

class CodeBERTClassifier(nn.Module):
    counter1=0 
    def __init__(self, model_path, num_labels=2, device='cuda'):        
        super(CodeBERTClassifier, self).__init__()        
        #self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        from transformers import RobertaTokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=num_labels,
        ignore_mismatched_sizes=True)
        self.block_size = 512
        self.embed = self.model.roberta.embeddings.word_embeddings
        self.vocab_size = self.embed.weight.size()[0]
        self.x_size = self.embed.weight.size()[-1]
        self.device = device
        self.max_len = 512  
        CodeBERTClassifier.counter = 0 
    def tokenize(self, inputs1, inputs2, cut_and_pad=False, ret_id=False):
    # def tokenize_pair(self, inputs1, inputs2, cut_and_pad=False, ret_id=False):
        rets = []
        if isinstance(inputs1, str):
            inputs1 = [inputs1]
            inputs2 = [inputs2]
        for s1, s2 in zip(inputs1, inputs2):
            if cut_and_pad:
                tokens_1 = self.tokenizer.tokenize(s1)[:self.block_size//2-2]
                tokens_2 = self.tokenizer.tokenize(s2)[:self.block_size//2-2]
                tokens = [self.tokenizer.cls_token] + tokens_1 + [self.tokenizer.sep_token] + tokens_2 + [self.tokenizer.sep_token]
                padding_length = self.block_size - len(tokens)
                tokens += [self.tokenizer.pad_token] * padding_length
            else:
                tokens_1 = self.tokenizer.tokenize(s1)
                tokens_2 = self.tokenizer.tokenize(s2)
                tokens = [self.tokenizer.cls_token] + tokens_1 + [self.tokenizer.sep_token] + tokens_2 + [self.tokenizer.sep_token]
            if not ret_id:
                rets.append(tokens)
            else:
                ids = self.tokenizer.convert_tokens_to_ids(tokens)
                rets.append(ids)
        return rets


    def run_batch(self, inputs_src1, inputs_src2, labels=None):
        inputs = self.tokenize(inputs_src1, inputs_src2, cut_and_pad=True, ret_id=True)
        inputs = torch.tensor(inputs, dtype=torch.long).to(self.device)
        if labels is not None:
            labels = torch.tensor(labels, dtype=torch.long).to(self.device)
        outputs = self.model(inputs, attention_mask=inputs.ne(self.tokenizer.pad_token_id), labels=labels)
        if labels is not None:
            return outputs.logits, outputs.loss
        else:
            return outputs.logits


    def forward(self, inputs1, inputs2, labels=None):       
        CodeBERTClassifier.counter += 1
        return self.run_batch(inputs1, inputs2, labels)

    def prob(self, inputs1, inputs2):       
        self.model.eval()
        with torch.no_grad():
            logits = self.forward(inputs1, inputs2)
        # logits = self.forward(inputs1, inputs2)
        prob = nn.Softmax(dim=-1)(logits)
        return prob
    def grad(self, inputs1, inputs2, labels):
        CodeBERTClassifier.counter += 1
        self.zero_grad()
        self.embed.weight.retain_grad()  # (vocab_size, hidden_dim)
        logits, loss = self.forward(inputs1, inputs2, labels)
        loss.backward()
        return self.embed.weight.grad


if __name__ == "__main__":       
    oj = OJ104(path="../data/oj.pkl.gz")
    model = CodeBERTClassifier('../codebert-base-mlm', 2).train()
    opt = optim.Adam(model.parameters(), lr=1e-5)    
    opt.zero_grad()    
    b = oj.train.next_batch(3)
    logits, loss = model.run_batch(b['x'], b['y'])    
    loss.backward()
    opt.step()
