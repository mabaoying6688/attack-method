# -*- coding: utf-8 -*-

import argparse
import os
from dataset import Author66
import torch
import torch.nn as nn
from torch import optim
import random
import numpy

    
def trainEpochs(epochs, training_set, valid_set, device,
                batch_size=32, print_each=10, plot_each=10, saving_path='./', patience=12):
    
    classifier.train()
    plot_losses = []
    epoch = 0
    i = 0
    print_loss_total = 0
    plot_loss_total = 0
    n_batch = int(training_set.get_size() / batch_size)
    best_acc = 0.0
    best_epoch = 0 
    patience_counter = 0
    print('start training epoch ' + str(epoch + 1) + '....')
    print(training_set.get_size(), batch_size, n_batch)
    while True:
        
        batch = training_set.next_batch(batch_size)
        if batch['new_epoch']:
            epoch += 1
            acc = evaluate(valid_set)
            classifier.train()
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch  
                patience_counter = 0  

                old_best = os.path.join(saving_path, "best_model")
                if os.path.exists(old_best):
                    import shutil
                    shutil.rmtree(old_best)

                # torch.save(classifier.state_dict(), best_model_path)
                best_model_path = os.path.join(saving_path, "best_model")
                classifier.model.save_pretrained(best_model_path)
                classifier.tokenizer.save_pretrained(best_model_path)
                print(f"[Info] Best model updated at epoch {epoch}. Acc: {best_acc:.4f}")

            if opt.lrdecay:
               adjust_learning_rate(optimizer)
            if patience_counter >= patience:
                print(f"[Early Stopping] No improvement for {patience} epochs. Stop training.")
                break

            if epoch == epochs:
                break
            i = 0
            print_loss_total = 0
            print('start training epoch ' + str(epoch + 1) + '....')

        optimizer.zero_grad()
        
        ouputs, loss = classifier(batch['x'], batch['y'])
        loss.backward()
        optimizer.step()

        print_loss_total += loss.item()

        if (i + 1) % print_each == 0: 
            print_loss_avg = print_loss_total / print_each
            print_loss_total = 0
            print('(%d %d%%) %.4f' % (epoch + 1, (i + 1) / n_batch * 100, print_loss_avg))
            
        i += 1
        

def adjust_learning_rate(optimizer, decay_rate=0.8):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
        
def evaluate(dataset, batch_size=128):
    
    classifier.eval()
    testnum = 0
    testcorrect = 0
    dataset.reset_epoch()
    batch = dataset.next_batch(batch_size)  
    
    while True:
        if batch.get('empty', False):  
            break
            
        with torch.no_grad():
            outputs = classifier(batch['x'])
            labels = torch.tensor(batch['y'], dtype=torch.long).to(device)
            preds = torch.argmax(outputs, dim=1)
            res = (preds == labels)
            testcorrect += torch.sum(res)
            testnum += len(labels)

        batch = dataset.next_batch(batch_size)
        if batch['new_epoch']:
            break
    
    if testnum == 0:
        print("[Error]")
        return 0
    
    acc = float(testcorrect) / testnum
    print(f'eval_acc: {acc:.4f}')
    return acc
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default="0")
    parser.add_argument('-lr', type=float, default=1e-5)
    parser.add_argument('-l2p', type=float, default=0.01)
    parser.add_argument('-lrdecay', action='store_true')
    parser.add_argument('--data', type=str, default="../data_author/author.pkl.gz")
    

    parser.add_argument('--pretrain', type=str, default="../codebert-base-mlm")

    parser.add_argument('--save_dir', type=str, default="../model/codebert_new")
    parser.add_argument('--bs', type=int, default=10)#batch size
    parser.add_argument('--nepoch', type=int, default=100)
    
    
    opt = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    if int(opt.gpu) < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    n_class = 66  

    batch_size = opt.bs
    n_epoch = opt.nepoch
    rand_seed = 1726

    torch.manual_seed(rand_seed)
    random.seed(rand_seed)
    numpy.random.seed(rand_seed)

    poj = Author66(path=opt.data)  

    training_set = poj.train
    valid_set = poj.dev
    test_set = poj.test

    print("Training set size:", poj.train.get_size())
    print("Validation set size:", poj.dev.get_size())
    print("Test set size:", poj.test.get_size())


    from codebert import CodeBERTClassifier
    
    classifier = CodeBERTClassifier(model_path=opt.pretrain,
                                    num_labels=n_class,
                                    device=device).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=0.00003)
    #optimizer = optim.Adam(classifier.parameters(), lr=opt.lr, weight_decay=opt.l2p)
    #optimizer = optim.SGD(classifier.parameters(), lr=0.1, weight_decay=0.5)
    trainEpochs(n_epoch, training_set, valid_set, device,
                saving_path=opt.save_dir, batch_size=batch_size)