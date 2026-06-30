# -*- coding: utf-8 -*-

from dataset import OJ104
from lstm_classifier import LSTMClassifier, LSTMEncoder, GRUClassifier, GRUEncoder
from modifier import TokenModifier, InsModifier
import random
import time
import torch
import argparse
import pickle, gzip
import os, sys
import pandas as pd
import numpy as np
import json  
voc_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../CARROT-main', 'preprocesslstm-MY', 'vocab_cnt_test.json') 
class CombinedAttacker(object):
    def __init__(self, dataset, symtab, instab, classifier):
        self.tokenM = TokenModifier(classifier=classifier,
                                    loss=torch.nn.CrossEntropyLoss(),
                                    uids=symtab['all'],
                                    txt2idx=dataset.get_txt2idx(),
                                    idx2txt=dataset.get_idx2txt())
        self.insM = InsModifier(classifier=classifier,
                                txt2idx=dataset.get_txt2idx(),
                                poses=None) # wait to init when attackself.cl = classifier#将传入的 classifier 参数赋值给 self.cl，表示分类器
        self.cl = classifier
        self.d = dataset
        self.inss = instab
        self.syms = symtab

    def attack(self,x_token,x, y, uids, poses, rand_d=None, vocab_cnt_test=None,n_candidate=100, n_iter=20):
     
        ins_success, ins_adv_x, ins_adv_y = self._attack_ins(x_token,x, y, poses, rand_d, n_candidate, n_iter)
        if ins_success:
            return True, ins_adv_x, ins_adv_y
        token_success, token_adv_x, token_adv_y = self._attack_token(x, y, uids, vocab_cnt_test, n_candidate, n_iter)
        if token_success:
             return True, token_adv_x, token_adv_y


        return False, x, y 
    def _attack_token(self, x, y, uids,vocab_cnt_test, n_candidate=100, n_iter=20):
        iter = 0
        n_stop = 0
        old_prob = self.cl.prob(torch.tensor(x, dtype=torch.long).cuda().permute([1, 0]))[0]
        if torch.argmax(old_prob) != y[0]:
            print ("SUCC! Original mistake.")
            return True, x, [torch.argmax(old_prob).cpu().numpy()]

        old_prob = old_prob[y[0]]
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
                new_x, new_uid_cand = self.tokenM.rename_uid(x, y, uids[k],vocab_cnt_test, k, n_candidate)
                new_prob = self.cl.prob(torch.tensor(new_x, dtype=torch.long).cuda().permute([1, 0]))
                new_pred = torch.argmax(new_prob, dim=1)
                for uid, p, pr, _x in zip(new_uid_cand, new_pred, new_prob, new_x):
                    if p != y[0]:
                        print ("SUCC!\t%s => %s\t\t%d(%.5f) => %d(%.5f) %d(%.5f)" % \
                               (k, self.d.idx2vocab(uid), y[0], old_prob, y[0], pr[y[0]], p, pr[p]))
                        return True, [_x], [p.cpu().numpy()]
    
                new_prob_idx = torch.argmin(new_prob[:, y[0]])
                if new_prob[new_prob_idx][y[0]] < old_prob:
                    x = [new_x[new_prob_idx]]
                    uids[self.d.idx2vocab(int(new_uid_cand[new_prob_idx]))] = uids.pop(k)
                    n_stop = 0
                    print ("acc\t%s => %s\t\t%d(%.5f) => %d(%.5f)" % \
                           (k, self.d.idx2vocab(int(new_uid_cand[new_prob_idx])),
                          y[0], old_prob, y[0], new_prob[new_prob_idx][y[0]]))
                    old_prob = new_prob[new_prob_idx][y[0]]
                else:
                    n_stop += 1
                    print ("rej\t%s" % k)
        print ("FAIL!")#
        return False, x, y
    
    def _attack_ins(self,x_token, x, y, poses,rand_d, n_candidate=100, n_iter=20):
        self.insM.initInsertDict(poses)

        iter = 0
        n_stop = 0
        old_prob = self.cl.prob(torch.tensor(x, dtype=torch.long).cuda().permute([1, 0]))[0]
        if torch.argmax(old_prob) != y[0]:
            #pass
            print ("SUCC! Original mistake.")
            return True, x, [torch.argmax(old_prob).cpu().numpy()]
        old_prob = old_prob[y[0]]
        while iter < n_iter:
            iter += 1
            n_could_del = self.insM.insertDict["count"]
            n_candidate_del = n_could_del
            n_candidate_ins = n_candidate - n_candidate_del
            assert n_candidate_del >= 0 and n_candidate_ins >= 0
            new_x_del, new_insertDict_del = self.insM.remove(x[0], n_candidate_del)
            new_x_add, new_insertDict_add = self.insM.insert4(x_token,x[0], rand_d, n_candidate_ins)
            new_x = new_x_del + new_x_add
            new_insertDict = new_insertDict_del + new_insertDict_add
            if new_x == []: # no valid candidates
                n_stop += 1
                continue
            feed_new_x = [_x[:self.cl.max_len] for _x in new_x] # this step is important
            feed_tensor = torch.tensor(feed_new_x, dtype=torch.long)
            new_prob = self.cl.prob(feed_tensor.cuda().permute([1, 0]))
            new_pred = torch.argmax(new_prob, dim=1)
            for insD, p, pr, _x in zip(new_insertDict, new_pred, new_prob, new_x):
                if p != y[0]:
                    print ("SUCC!\tinsert_n %d => %d\t\t%d(%.5f) => %d(%.5f) %d(%.5f)" % \
                            (self.insM.insertDict["count"], insD["count"], 
                                y[0], old_prob, y[0], pr[y[0]], p, pr[p]))
                    return True, [_x], [p.cpu().numpy()]

            new_prob_idx = torch.argmin(new_prob[:, y[0]])
            if new_prob[new_prob_idx][y[0]] < old_prob:
                print ("acc\tinsert_n %d => %d\t\t%d(%.5f) => %d(%.5f)" % \
                        (self.insM.insertDict["count"], new_insertDict[new_prob_idx]["count"], 
                          y[0], old_prob, y[0], new_prob[new_prob_idx][y[0]]))
                self.insM.insertDict = new_insertDict[new_prob_idx] # don't forget this step
                n_stop = 0
                old_prob = new_prob[new_prob_idx][y[0]]
            else:
                n_stop += 1
                print ("rej\t%s" % "")
            if n_stop >= len(new_x):    # len(new_x) could be smaller than n_candidate
                iter = n_iter
                break
        print ("FAIL!")
        return False, x, y
    def attack_all(self, n_candidate=100, n_iter=20, dump_samples_path=None):        
        n_succ = 0
        total_time = 0
        st_time = time.time()
        with open('../preprocesslstm/vocab_cnt_test.json', 'r') as VOCA:  
             vocab_cnt_test = json.load(VOCA)  
        for i in range(self.d.test.get_size()):
            b = self.d.test.next_batch(1)
            print ("\t%d/%d\tID = %d\tY = %d" % (i+1, self.d.test.get_size(), b['id'][0], b['y'][0]))
            start_time = time.time()
            # Generate rand_d for ins attack
            rand_d = []
            for _ in range(30):
                rand = self.d.test.next_batch(1)
                rand_d.append(rand)
            tag, adv_x, adv_y = self.attack(b['raw'],b['x'], b['y'], self.syms['te'][b['id'][0]], self.inss['stmt_te'][b['id'][0]], rand_d, vocab_cnt_test,n_candidate, n_iter)
            #print('tag=',tag)
            if tag:
                n_succ += 1
                total_time += time.time() - start_time
            if n_succ <= 0:
                pass
                flush=True 
                print ("\tCurr succ rate = %.3f, Avg time cost = NaN sec" \
                       % (n_succ/(i+1)), flush=True)
            else:
                pass
                print ("\tCurr succ rate = %.3f, Avg time cost = %.1f sec, Call times = %d " \
                       % (n_succ/(i+1), total_time/n_succ, (GRUClassifier.counter+GRUEncoder.counter1)), flush=True)                
        print("[Task Done] Time Cost: %.1f sec Succ Rate: %.3f Aver Call times: %.3f" % (time.time()-st_time, n_succ/self.d.test.get_size(),(GRUClassifier.counter+GRUEncoder.counter1)/n_succ))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default="1")
    parser.add_argument('--attn', action='store_true')
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    device = torch.device("cuda")

    vocab_size = 5000

    embedding_size = 512
    hidden_size = 600
    n_layers = 2
    num_classes = 104##########
    max_len = 600

    poj = OJ104(path="../data/oj.pkl.gz",
                max_len=max_len,
                vocab_size=vocab_size)
    training_set = poj.train
    #print(poj.train)
    valid_set = poj.dev
    test_set = poj.test

    with gzip.open('../data/oj_uid.pkl.gz', "rb") as f:
        symtab = pickle.load(f)
    with gzip.open('../data/oj_inspos.pkl.gz', "rb") as f:
        instab = pickle.load(f)
    enc = GRUEncoder(embedding_size, hidden_size, n_layers)
    classifier = GRUClassifier(vocab_size, embedding_size, enc,
                                hidden_size, num_classes, max_len, attn=opt.attn).cuda()

    classifier.load_state_dict(torch.load('../code-OJ-GRU/model10.pt'))
    attacker = CombinedAttacker(poj, symtab, instab, classifier)
    attacker.attack_all(10, 10)

    