# -*- coding: utf-8 -*-

import argparse
import sys
import os
from dataset import OJ104
from sklearn.metrics import precision_recall_fscore_support
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import random
import numpy   
import transformers
transformers.logging.set_verbosity(transformers.logging.ERROR)   
    
def gettensor(batch, batchfirst=False):
    
    x1, x2, labels = batch['x1'], batch['x2'], batch['y']
    x1, x2, labels = torch.tensor(x1, dtype=torch.long).cuda(), \
                     torch.tensor(x2, dtype=torch.long).cuda(), \
                     torch.tensor(labels, dtype=torch.long).cuda()
    if batchfirst:
#         inputs_pos = [[pos_i + 1 if w_i != 0 else 0 for pos_i, w_i in enumerate(inst)] for inst in inputs]
#         inputs_pos = torch.tensor(inputs_pos, dtype=torch.long).cuda()
        return x1, x2, labels
    x1 = x1.permute([1, 0])
    x2 = x2.permute([1, 0])
    return x1, x2, labels
     
def trainEpochs(epochs, training_set, valid_set, batch_size=32, print_each=100, plot_each=100, saving_path='./'):    
    classifier.train()
    plot_losses = []   
    epoch = 0
    i = 0
    print_loss_total = 0
    plot_loss_total = 0
    n_batch = int(training_set.get_size() / batch_size)    
    print('start training epoch ' + str(epoch + 1) + '....')    
    while True:
        
        batch = training_set.next_batch(batch_size)
        if batch['new_epoch']:
            epoch += 1
            evaluate(valid_set)
            classifier.train()
            # torch.save(classifier.state_dict(), saving_path + str(epoch) + '.pt')
            classifier.model.save_pretrained(os.path.join(saving_path, str(epoch) + ".pt"))
            classifier.tokenizer.save_pretrained(os.path.join(saving_path, str(epoch) + ".pt"))#原始有
            
            if opt.lrdecay:
                adjust_learning_rate(optimizer)
            
            if epoch == epochs:
                break
            i = 0
            print('start training epoch ' + str(epoch + 1) + '....')

        # x1, x2, labels = gettensor(batch, batchfirst=(_model == 'Transformer'))

        labels = torch.tensor(batch['y'], dtype=torch.long).cuda()
        batch = training_set.next_batch(batch_size)
        inputs_src1 = [" ".join(tokens) for tokens in batch['raw1']]  # 转为字符串
        inputs_src2 = [" ".join(tokens) for tokens in batch['raw2']]
        labels = batch['y']
        optimizer.zero_grad()
        outputs, loss = classifier(inputs_src1, inputs_src2, labels=labels) 
        loss.backward()
        optimizer.step()

        print_loss_total += loss.item()
        plot_loss_total += loss.item()

        if (i + 1) % print_each == 0: 
            print_loss_avg = print_loss_total / print_each
            print_loss_total = 0
            print('(%d %d%%) %.4f' % (epoch + 1, (i + 1) / n_batch * 100, print_loss_avg), flush=True)
        if (i + 1) % plot_each == 0:
            plot_loss_avg = plot_loss_total / plot_each
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0             
        i += 1
        

def adjust_learning_rate(optimizer, decay_rate=0.8):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
        
            
def evaluate(dataset, batch_size=128):
    all_preds = []
    all_labels = []
    classifier.eval()
    testnum = 0
    testcorrect = 0
    
    while True:
        
        batch = dataset.next_batch(batch_size)
        if batch['new_epoch']:
            break
        labels = torch.tensor(batch['y'], dtype=torch.long).cuda()
        x1 = [" ".join(tokens) for tokens in batch['raw1']]  # 转为字符串
        x2 = [" ".join(tokens) for tokens in batch['raw2']]
        labels = torch.tensor(batch['y'], dtype=torch.long).cuda()
        with torch.no_grad():
            outputs = classifier(x1, x2)
            preds = torch.argmax(outputs, dim=1)
            res = torch.argmax(outputs, dim=1) == labels
            testcorrect += torch.sum(res)
            testnum += len(labels)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    print('eval_acc:  %.2f' % (float(testcorrect) * 100.0 / testnum))
    print(f'Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}')
       
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', default="1")
    parser.add_argument('-lr', type=float, default=0.00003)
    parser.add_argument('-save_dir', default="../model/model1")
    parser.add_argument('--pretrain', type=str, default="../codebert-base-mlm")
    parser.add_argument('-l2p', type=float, default="0")
    parser.add_argument('-lrdecay', action='store_true')
    parser.add_argument('--data', type=str, default='../data_clone/data_clone/oj.pkl.gz')
    # parser.add_argument('--adv_train_path', type=str, default=None)
    # parser.add_argument('--adv_train_size', type=int, default=None)
    parser.add_argument('--bs', type=int, default=10)#batch size
    parser.add_argument('--nepoch', type=int, default=15)
    
    opt = parser.parse_args()
    
    # _model = opt.model
    n_class = 2 
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    device = torch.device("cuda")


    n_classe = 2
    max_len = 512
    rand_seed = 1726
    
    batch_size = opt.bs
    n_epoch = opt.nepoch
    torch.manual_seed(rand_seed)
    random.seed(rand_seed)
    numpy.random.seed(rand_seed)
    poj = OJ104(path=opt.data,
                max_len=max_len
                # ,vocab_size=vocab_size,
                # adv_train_path=opt.adv_train_path,
                # adv_train_size=opt.adv_train_size
                )   
    training_set = poj.train
    valid_set = poj.dev
    test_set = poj.test
    
    # import transformers after gpu selection
    from codebert import CodeBERTClassifier
    
    classifier = CodeBERTClassifier(model_path=opt.pretrain,
                                    num_labels=n_class,
                                    device=device).to(device)
    
    optimizer = optim.Adam(classifier.parameters(), lr=0.00003)
    #optimizer = optim.Adam(classifier.parameters(), lr=opt.lr, weight_decay=opt.l2p)#原来
    #optimizer = optim.SGD(classifier.parameters(), lr=0.1, weight_decay=0.5)
    trainEpochs(15, training_set, valid_set, saving_path=opt.save_dir)
    # evaluate(test_set)