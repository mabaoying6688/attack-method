import argparse
import sys
import os
from dataset import Author66
from lstm_classifier import LSTMClassifier, LSTMEncoder, GRUClassifier, GRUEncoder

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def gettensor(batch, batchfirst=False):
    inputs, labels = batch['x'], batch['y']
    inputs, labels = torch.tensor(inputs, dtype=torch.long).cuda(), \
                     torch.tensor(labels, dtype=torch.long).cuda()
    if batchfirst:
        return inputs, labels
    inputs = inputs.permute([1, 0])
    return inputs, labels


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def trainEpochs(epochs, training_set, valid_set, batch_size=64, print_each=10, saving_path='./', patience=30):
    classifier.train()
    plot_losses = []
    epoch = 0
    i = 0
    print_loss_total = 0
    plot_loss_total = 0
    n_batch = int(training_set.get_size() / batch_size)

    best_acc = 0.0
    best_val_loss = float('inf')
    best_model_path = os.path.join(saving_path, 'best_model.pt')
    patience_counter = 0
    print('start training epoch ' + str(epoch + 1) + '....')
    print(training_set.get_size(), batch_size, n_batch)

    accumulation_steps = 2

    while True:
        batch = training_set.next_batch(batch_size)
        if batch['new_epoch']:
            epoch += 1
            acc, val_loss, f1, precision, recall = evaluate(valid_set, batch_size)
            classifier.train()
            
            if acc > best_acc:
                best_acc = acc
                best_val_loss = val_loss
                torch.save(classifier.state_dict(), best_model_path)
                print(f"[Info] Best model updated. New best acc: {best_acc:.4f}, val_loss: {val_loss:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"[Info] No improvement. Patience counter: {patience_counter}/{patience}")
            

            if patience_counter >= patience:
                print(f"[Early Stopping] No improvement for {patience} epochs. Stop training.")
                break
            if epoch == epochs:
                break
            i = 0
            print('start training epoch ' + str(epoch + 1) + '....')
        if i % accumulation_steps == 0:
            optimizer.zero_grad()

        inputs, labels = gettensor(batch, batchfirst=(_model == 'Transformer'))
        outputs = classifier(inputs)[0]
        loss = criterion(outputs, labels) / accumulation_steps
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        print_loss_total += loss.item() * accumulation_steps
        plot_loss_total += loss.item() * accumulation_steps

        if (i + 1) % print_each == 0:
            print_loss_avg = print_loss_total / print_each
            print_loss_total = 0
            current_lr = scheduler.get_last_lr()[0]
            print('(%d %d%%) loss: %.4f, lr: %.6f' % (epoch + 1, (i + 1) / n_batch * 100, print_loss_avg, current_lr))
        
        if (i + 1) % (print_each * 5) == 0:
            plot_loss_avg = plot_loss_total / (print_each * 5)
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        i += 1

def evaluate(dataset, batch_size=64):
    classifier.eval()
    testnum = 0
    testcorrect = 0
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        while True:
            batch = dataset.next_batch(batch_size)
            if batch['new_epoch']:
                break
            inputs, labels = gettensor(batch, batchfirst=(_model == 'Transformer'))
            outputs = classifier(inputs)[0]
            
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item() * len(labels)
            
            preds = torch.argmax(outputs, dim=1)
            res = preds == labels
            testcorrect += torch.sum(res).item()
            testnum += len(labels)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    if testnum == 0:
        print("[Warning] Empty evaluation dataset!")
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    acc = float(testcorrect) / testnum
    avg_loss = total_loss / testnum
    
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    
    print('eval_acc: %.4f, loss: %.4f, F1: %.4f, Precision: %.4f, Recall: %.4f' % 
          (acc, avg_loss, f1, precision, recall))
    return acc, avg_loss, f1, precision, recall


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default="0")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', default="../model/author_lstm")
    parser.add_argument('--save_name', default="author_LSTM")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--attn', action='store_true', default=True)
    parser.add_argument('--model', default="LSTM")
    
    parser.add_argument('--l2p', type=float, default=1e-4)
    parser.add_argument('--data', type=str, default='../data_author/author.pkl.gz')
    parser.add_argument('--adv_train_path', type=str, default=None)
    parser.add_argument('--adv_train_size', type=int, default=20)
    opt = parser.parse_args()

    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)

    _model = opt.model
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vocab_size = 2300
    embedding_size = 512
    hidden_size = 1024
    n_layers = 2
    num_classes = 66
    max_len = 500

    poj = Author66(path=opt.data,
                   max_len=max_len,
                   vocab_size=vocab_size,
                   adv_train_path=opt.adv_train_path,
                   adv_train_size=opt.adv_train_size)
    training_set = poj.train
    valid_set = poj.dev
    test_set = poj.test

    if _model == 'LSTM':
        enc = LSTMEncoder(embedding_size, hidden_size, n_layers, drop_prob=opt.dropout, brnn=True)
        classifier = LSTMClassifier(vocab_size, embedding_size, enc, hidden_size, num_classes, max_len,
                                     dropout_p=opt.dropout, attn=opt.attn).to(device)
    elif _model == 'GRU':
        enc = GRUEncoder(embedding_size, hidden_size, n_layers, drop_prob=opt.dropout, brnn=True)
        classifier = GRUClassifier(vocab_size, embedding_size, enc, hidden_size, num_classes, max_len,
                                   dropout_p=opt.dropout, attn=opt.attn).to(device)
    elif _model == 'Transformer':
        print("Transformer model not implemented yet")
        exit()

    optimizer = optim.AdamW([
        {'params': classifier.embedding.parameters(), 'lr': 1e-4},
        {'params': classifier.encoder.parameters(), 'lr': 5e-4},
        {'params': classifier.classifier.parameters(), 'lr': 1e-3}
    ], weight_decay=opt.l2p)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )

    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    os.makedirs(opt.save_dir, exist_ok=True)

    print(f"Starting training with {opt.model} model")
    print(f"Vocabulary size: {vocab_size}, Embedding size: {embedding_size}")
    print(f"Hidden size: {hidden_size}, Num classes: {num_classes}")

    trainEpochs(opt.epoch, training_set, valid_set, batch_size=opt.batch_size, saving_path=opt.save_dir)

    print('\nFinal evaluation on test set...')

    best_model_path = os.path.join(opt.save_dir, 'best_model.pt')
    classifier.load_state_dict(torch.load(best_model_path))
    evaluate(test_set, batch_size=opt.batch_size)