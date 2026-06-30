#-*- coding: utf-8 -*-

import argparse
import sys
import os
from dataset import Author66
from lscnn import LSCNNClassifier

import torch
import torch.nn as nn
from torch import optim
import random
import numpy

def gettensor(batch, device):
    return (torch.tensor(batch['x'], dtype=torch.long).to(device),
            torch.tensor(batch['l'], dtype=torch.long).to(device),
            torch.tensor(batch['y'], dtype=torch.long).to(device))

def trainEpochs(epochs, training_set, valid_set, device,
               batch_size=32, print_each=10, plot_each=10, saving_path='./',patience=70):

    classifier.train()
    plot_losses = []
    epoch = 0
    i = 0
    print_loss_total = 0
    plot_loss_total = 0

    n_batch = int(training_set.get_size() / batch_size)

    best_acc = 0.0
    best_model_path = os.path.join(saving_path, 'best_model.pt')
    patience_counter = 0
    print('start training epoch ' + str(epoch + 1) + '....')
    print(training_set.get_size(), batch_size, n_batch)

    step = 0
    while True:
        batch = training_set.next_batch(batch_size)
        inputs, lens, labels = gettensor(batch, device)
        step += 1
        
        if batch['new_epoch']:
            epoch += 1
            acc = evaluate(valid_set, device, batch_size)
            classifier.train()
            if acc > best_acc:
                best_acc = acc
                torch.save(classifier.state_dict(), best_model_path)
                print(f"[Info] Best model updated. New best acc: {best_acc:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"[Early Stopping] No improvement for {patience} epochs. Stop training.")
                break
            if epoch == epochs:
                break
            i = 0
            print('start training epoch ' + str(epoch + 1) + '....')

        warmup_lr(optimizer, step)

        optimizer.zero_grad()
        outputs = classifier(inputs, lens)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), 2.0)
        optimizer.step()

        print_loss_total += loss.item()
        plot_loss_total += loss.item()

        if (i + 1) % print_each == 0:
            print_loss_avg = print_loss_total / print_each
            print_loss_total = 0
            print('(%d %d%%) %.4f' % (epoch + 1, (i + 1) / n_batch * 100, print_loss_avg))
        if (i + 1) % plot_each == 0:
            plot_loss_avg = plot_loss_total / plot_each
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
        i += 1

def warmup_lr(optimizer, step, warmup_steps=400):
    for pg in optimizer.param_groups:
        pg['lr'] = opt.lr * min(step / warmup_steps, 1.0)

def adjust_learning_rate(optimizer, decay_rate=0.8):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
        
def evaluate(dataset, device, batch_size=128):
    classifier.eval()
    testnum = 0
    testcorrect = 0   
    with torch.no_grad():
        while True:
            batch = dataset.next_batch(batch_size)
            if batch['new_epoch']:
                break
            inputs, lens, labels = gettensor(batch, device)
            outputs = classifier(inputs, lens)
            outputs = torch.softmax(outputs, dim=-1)
            if torch.isnan(outputs).any() or torch.isinf(outputs).any() or (outputs.max() > 1e5) or (outputs.min() < -1e5):
                print("Warning: skipping abnormal batch")
                continue
            res = torch.argmax(outputs, dim=1) == labels
            testcorrect += torch.sum(res)
            testnum += len(labels)
    acc = 0.0
    if testnum > 0:
        acc = float(testcorrect) / testnum
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default="0")
    parser.add_argument('-lr', type=float, default=5e-5)
    parser.add_argument('-l2p', type=float, default=1e-3)
    parser.add_argument('-lrdecay', action='store_true')
    parser.add_argument('--data', type=str, default="../data_author/author.pkl.gz")
    parser.add_argument('--save_dir', type=str, default="../model/author-lscnn")
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--nepoch', type=int, default=100)    
    global opt
    opt = parser.parse_args()   
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    if int(opt.gpu) < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    n_class = 66
    vocab_size = 2300
    embed_width = 512
    n_conv = 4096
    conv_size = 7
    lstm_size = 300
    n_lstm = 1
    max_stmt_cnt = 40
    max_stmt_len = 20
    brnn = True
    batch_size = opt.bs
    n_epoch = opt.nepoch
    rand_seed = 1726   
    torch.manual_seed(rand_seed)
    random.seed(rand_seed)
    numpy.random.seed(rand_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    poj = Author66(path=opt.data,
                max_stmt_len=max_stmt_len,
                max_stmt_cnt=max_stmt_cnt,
                vocab_size=vocab_size)
    training_set = poj.train
    valid_set = poj.dev
    test_set = poj.test
    classifier = LSCNNClassifier(n_class, vocab_size, embed_width,
                                 n_conv, conv_size, lstm_size, n_lstm,
                                 brnn, device).to(device)   
    optimizer = optim.AdamW(classifier.parameters(), lr=opt.lr, weight_decay=opt.l2p)
    criterion = nn.CrossEntropyLoss()   
    trainEpochs(n_epoch, training_set, valid_set, device,
                saving_path=opt.save_dir, batch_size=batch_size)
