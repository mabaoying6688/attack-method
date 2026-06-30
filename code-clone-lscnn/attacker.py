# -*- coding: utf-8 -*-

import time
from sklearn.metrics import confusion_matrix
from dataset import OJ104
from lscnn import LSCNNClassifier
from modifier import TokenModifier, InsModifier
from modifier import get_batched_data, gettensor
import pandas as pd
import numpy as np
import numpy
import random
import torch
import torch.nn as nn
import argparse
import pickle, gzip
import os, sys, time
from sklearn.metrics import precision_recall_fscore_support
import json  


class CombinedAttacker(object):
    def __init__(self, dataset, symtab, instab, classifier):
        self.txt2idx = dataset.get_txt2idx()
        self.idx2txt = dataset.get_idx2txt()
        self.max_stmt_cnt = dataset.get_max_stmt_cnt()
        self.max_stmt_len = dataset.get_max_stmt_len()
        self.tokenM = TokenModifier(classifier=classifier,
                                    loss=torch.nn.CrossEntropyLoss(),
                                    uids=symtab['all'],
                                    txt2idx=self.txt2idx,
                                    idx2txt=self.idx2txt,
                                    max_stmt_cnt=self.max_stmt_cnt,
                                    max_stmt_len=self.max_stmt_len)
        self.insM = InsModifier(classifier=classifier,
                                txt2idx=self.txt2idx,
                                idx2txt=self.idx2txt,
                                poses=None) # wait to init when attack
        self.cl = classifier
        self.d = dataset
        self.syms = symtab
        self.txt2idx = dataset.get_txt2idx()
        self.idx2txt = dataset.get_idx2txt()
        self.max_stmt_cnt = dataset.get_max_stmt_cnt()
        self.max_stmt_len = dataset.get_max_stmt_len()
        self.inss = instab
    def attack(self,x, x2, y, uids,poses,rand_d,vocab_cnt_test=None,n_candidate=100, n_iter=20,relax=1):
        # First, perform ins attack
        x_token=x
        ins_success, ins_adv_y = self._attack_ins(x_token,x,x2, y, poses, rand_d, n_candidate, n_iter,relax=1)
        if ins_success:
            return True, ins_adv_y

        token_success,  token_adv_y = self._attack_token(x,x2, y, uids, vocab_cnt_test, n_candidate, n_iter)
        if token_success:
            return True, token_adv_y
        return False, y
   
    def _attack_token(self, raw1, raw2, y, uids,vocab_cnt_test, n_candidate=100, n_iter=20):       
        iter = 0
        n_stop = 0

        batch = get_batched_data([raw1], [raw2], [y], self.txt2idx, self.max_stmt_cnt, self.max_stmt_len, self.cl.vocab_size)
        inputs1, inputs2, lens1, lens2, labels = gettensor(batch, self.cl.device)
        old_prob = self.cl.prob(inputs1, inputs2, lens1, lens2)[0]
        if torch.argmax(old_prob) != y:
            print ("SUCC! Original mistake.")
            return True, torch.argmax(old_prob).cpu().numpy()
        old_prob = old_prob[y]
        while iter < n_iter:
            keys = list(uids.keys())
            for k in keys:
                if iter >= n_iter:
                    break
                if n_stop >= len(uids):
                    iter = n_iter
                    break
                if k in self.tokenM.forbidden_uid:
                    n_stop += 1
                    continue
                iter += 1
                new_raw1, new_raw2, new_x_uid = self.tokenM.rename_uid(raw1, raw2, y,vocab_cnt_test, k, n_candidate)
                if new_raw1 is None:
                    n_stop += 1
                    print ("skip unk\t%s" % k)
                    continue
                batch = get_batched_data(new_raw1, new_raw2, [y]*len(new_raw1), self.txt2idx, 
                                         self.max_stmt_cnt, self.max_stmt_len, self.cl.vocab_size)
                inputs1, inputs2, lens1, lens2, labels = gettensor(batch, self.cl.device)
                new_prob = self.cl.prob(inputs1, inputs2, lens1, lens2)
                new_pred = torch.argmax(new_prob, dim=-1)
                for uid, p, pr in zip(new_x_uid, new_pred, new_prob):
                    if p != y:
                        print ("SUCC!\t%s => %s\t\t%d(%.5f) => %d(%.5f) %d(%.5f)" % \
                               (k, self.idx2txt[uid], y, old_prob, y, pr[y], p, pr[p]))
                        return True, p
                    
                new_prob_idx = torch.argmin(new_prob[:, y])
                if new_prob[new_prob_idx][y] < old_prob:
                    raw1 = new_raw1[new_prob_idx]
                    uids[self.idx2txt[int(new_x_uid[new_prob_idx])]] = uids.pop(k)
                    n_stop = 0
                    print ("acc\t%s => %s\t\t%d(%.5f) => %d(%.5f)" % \
                           (k, self.idx2txt[int(new_x_uid[new_prob_idx])],
                           y, old_prob, y, new_prob[new_prob_idx][y]))
                    old_prob = new_prob[new_prob_idx][y]
                else:
                    n_stop += 1
                    print ("rej\t%s" % k)
        print ("FAIL!")
        return False, y
    def _attack_ins(self,x_token, raw1, raw2, y, poses,rand_d, n_candidate=100, n_iter=20,relax=1):    
        self.insM.initInsertDict(poses)
        iter = 0
        n_stop = 0
        batch = get_batched_data([raw1], [raw2], [y], self.txt2idx, 
                                 self.max_stmt_cnt, self.max_stmt_len, self.cl.vocab_size)
        inputs1, inputs2, lens1, lens2, labels = gettensor(batch, self.cl.device)
        old_prob = self.cl.prob(inputs1, inputs2, lens1, lens2)[0]
        if torch.argmax(old_prob) != y:
            print ("SUCC! Original mistake.")
            return True, torch.argmax(old_prob).cpu().numpy()
        old_prob = old_prob[y]
        while iter < n_iter:
            iter += 1
            # get insertion candidates
            n_could_del = self.insM.insertDict["count"]
            n_candidate_del = n_could_del
            n_candidate_ins = n_candidate - n_candidate_del
            assert n_candidate_del >= 0 and n_candidate_ins >= 0
            new_raw1_del, new_insertDict_del = self.insM.remove(raw1, n_candidate_del)
            new_raw1_add, new_insertDict_add = self.insM.insert4(x_token,raw1,raw2,rand_d, n_candidate_ins)
            new_raw1 = new_raw1_del + new_raw1_add
            new_insertDict = new_insertDict_del + new_insertDict_add
            if new_raw1 == []: # no valid candidates
                n_stop += 1
                continue
            # find if there is any candidate successful wrong classfied
            batch = get_batched_data(new_raw1, [raw2]*len(new_raw1), [y]*len(new_raw1), self.txt2idx, 
                                     self.max_stmt_cnt, self.max_stmt_len, self.cl.vocab_size)
            inputs1, inputs2, lens1, lens2, labels = gettensor(batch, self.cl.device)
            new_prob = self.cl.prob(inputs1, inputs2, lens1, lens2)
            new_pred = torch.argmax(new_prob, dim=-1)
            for insD, p, pr in zip(new_insertDict, new_pred, new_prob):
                if p != y:
                    print ("SUCC!\tinsert_n %d => %d\t\t%d(%.5f) => %d(%.5f) %d(%.5f)" % \
                            (self.insM.insertDict["count"], insD["count"], 
                                y, old_prob, y, pr[y], p, pr[p]))
                    return True, p

            # if not, get the one with the lowest target_label_loss
            new_prob_idx = torch.argmin(new_prob[:, y])
            if new_prob[new_prob_idx][y] < old_prob:
                print ("acc\tinsert_n %d => %d\t\t%d(%.5f) => %d(%.5f)" % \
                        (self.insM.insertDict["count"], new_insertDict[new_prob_idx]["count"], 
                        y, old_prob, y, new_prob[new_prob_idx][y]))
                self.insM.insertDict = new_insertDict[new_prob_idx] # don't forget this step
                n_stop = 0
                old_prob = new_prob[new_prob_idx][y]
            else:
                n_stop += 1
                print ("rej\t%s" % "")
            if n_stop >= len(new_raw1):    # len(new_x) could be smaller than n_candidate
                iter = n_iter
                break
        print ("FAIL!")
        return False, y
    def attack_all(self, n_candidate=100, n_iter=20, relax=1, res_save=None):        
        n_succ = 0
        total_time = 0
        trues = []
        preds = []
        st_time = time.time()
        with open('../vocab_cnt_test.json', 'r') as VOCA:  
            vocab_cnt_test = json.load(VOCA)  
        loss_df = pd.DataFrame(columns=['test_num', 'n_succ','succ rate','call times', 'Aver call times'])
        for i in range(self.d.test.get_size()):
            b = self.d.test.next_batch(1)
            print ("\t%d/%d\tID = (%d, %d)\tY = %d" % (i+1, self.d.test.get_size(), b['id1'][0], b['id2'][0], b['y'][0]))
            start_time = time.time()
        # Generate rand_d for ins attack
            rand_d = []
            for _ in range(30):
                rand = self.d.test.next_batch(1)
                rand_d.append(rand)
            # print("rand_d",b['raw1'])
            # print("attackall:raw1:",b['raw1'])
            tag, pred = self.attack(b['raw1'][0],b['raw2'][0], b['y'][0], self.syms['te'][b['id1'][0]], self.inss['stmt_te'][b['id1'][0]],
                                           rand_d,vocab_cnt_test,n_candidate, n_iter, relax)
            if tag==True:
                n_succ += 1
                total_time += time.time() - start_time
            preds.append(int(pred))
            trues.append(int(b['y'][0]))                    
            if n_succ <= 0:
                print ("\tCurr succ rate = %.3f, Avg time cost = NaN sec" \
                       % (n_succ/(i+1)), flush=True)
            else:
                print ("\tCurr succ rate = %.3f, Avg time cost = %.1f sec, Call times = %d " \
                       % (n_succ/(i+1), total_time/n_succ, LSCNNClassifier.counter), flush=True)
                new_data = pd.DataFrame({
                    'test_num': [i + 1],
                     'n_succ': [n_succ],
                    'succ rate': [n_succ/(i+1)] , 
                    'call times':[LSCNNClassifier.counter], 
                    'Aver call times':[LSCNNClassifier.counter/n_succ]
                    })
                loss_df = pd.concat([loss_df, new_data], ignore_index=True)
                loss_df.to_csv('2025-3MHM-lscnn-clone-succ_values20-30.csv', index=False)  
            precision, recall, f1, _ = precision_recall_fscore_support(trues, preds, average='binary')
            print("\t(P, R, F1) = (%.3f, %.3f, %.3f)" % (precision, recall, f1))
        print("[Task Done] Time Cost: %.1f sec Succ Rate: %.3f Aver Call times: %.3f" % (time.time()-st_time, n_succ/self.d.test.get_size(),LSCNNClassifier.counter/n_succ))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default="1")
    parser.add_argument('--data', type=str, default="../data_clone/oj.pkl.gz")
    parser.add_argument('--model_path', type=str, default="../model/DWA/20.pt")
    parser.add_argument('--bs', type=int, default=16)
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    if int(opt.gpu) < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    torch.cuda.empty_cache()
    n_class = 2
    vocab_size = 2000
    embed_width = 512
    n_conv = 300
    conv_size = 5
    lstm_size = 400
    n_lstm = 1
    max_stmt_cnt = 40
    max_stmt_len = 20
    # max_stmt_cnt = 40
    # max_stmt_len = 20
    brnn = True

    batch_size = opt.bs
    rand_seed = 1726

    torch.manual_seed(rand_seed)
    random.seed(rand_seed)
    numpy.random.seed(rand_seed)

    poj = OJ104(path=opt.data,
                max_stmt_len=max_stmt_len,
                max_stmt_cnt=max_stmt_cnt,
                vocab_size=vocab_size)
    training_set = poj.train
    valid_set = poj.dev
    test_set = poj.test

    with gzip.open('../data_clone/oj_uid.pkl.gz', "rb") as f:
        symtab = pickle.load(f)
    with gzip.open('../data_clone/oj_inspos.pkl.gz', "rb") as f:
        instab = pickle.load(f)
        
    classifier = LSCNNClassifier(n_class, vocab_size, embed_width,
                                 n_conv, conv_size, lstm_size, n_lstm,
                                 brnn, device).to(device)

    classifier.load_state_dict(torch.load(opt.model_path))
    classifier.device = device

    attacker = CombinedAttacker(poj, symtab, instab, classifier)
    attacker.attack_all(20,30)

