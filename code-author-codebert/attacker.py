# -*- coding: utf-8 -*-
import json
import pandas as pd
from dataset import Author66
from modifier import TokenModifier, InsModifier
from modifier import get_batched_data
import numpy as np
import numpy
import torch
import torch.nn as nn
import argparse
import pickle, gzip
import os, sys, time
import pattern
import gc
import random


class CombinedAttacker(object):
    
    def __init__(self, dataset, symtab, instab, classifier):
        self.txt2idx = dataset.get_txt2idx()
        self.idx2txt = dataset.get_idx2txt()
        self.tokenM = TokenModifier(classifier=classifier,
                                    loss=torch.nn.CrossEntropyLoss(),
                                    uids=symtab['all'],
                                    txt2idx=dataset.get_txt2idx(),
                                    idx2txt=dataset.get_idx2txt())
        self.insM = InsModifier(classifier=classifier,
                                txt2idx=dataset.get_txt2idx(),
                                
                                idx2txt=dataset.get_idx2txt(),
                                poses=None)  # wait to init when attack
        self.cl = classifier
        self.d = dataset
        self.syms = symtab
        self.inss = instab
    
    def attack(self,x_token,x, y, uids, poses, rand_d=None, vocab_cnt_test=None,n_candidate=100, n_iter=20):
        # x = [item for sublist in x_token for item in sublist]
        x =x_token 
        ins_success = self._attack_ins(x_token,x, y, poses, rand_d, n_candidate, n_iter)
        if ins_success:
            return True
        token_success= self._attack_token(x, y, uids, vocab_cnt_test, n_candidate, n_iter)
        if token_success:
            return True            
        return False
    
    def _attack_token(self, x, y, uids, vocab_cnt_test, n_candidate=100, n_iter=20):
        iter = 0
        n_stop = 0
        batch = get_batched_data([x], [y], self.txt2idx)
        old_prob = self.cl.prob(batch['x'])[0]
        # print("torch.argmax(old_prob)=",torch.argmax(old_prob))
        # print("y=",y)
        if torch.argmax(old_prob) != y:
            print("SUCC! Original mistake.")
            return True

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
                assert not k.startswith('Ġ')
                Gk = 'Ġ' + k
                Gk_idx = self.cl.tokenizer.convert_tokens_to_ids(Gk)
                if Gk_idx == self.cl.tokenizer.unk_token_id:
                    continue
                iter += 1
                # print("x=",x)
                new_x, new_uid_cand = self.tokenM.rename_uid(uids, x, y, uids[k], vocab_cnt_test, k, n_candidate)
                if new_x is None:
                    n_stop += 1
                    print ("skip unk\t%s" % k)
                    continue
                batch = get_batched_data(new_x, [y]*len(new_x), self.txt2idx)
                
                new_prob = self.cl.prob(batch['x'])
                new_pred = torch.argmax(new_prob, dim=-1)
                for uid, p, pr, _x in zip(new_uid_cand, new_pred, new_prob, new_x):
                    if p != y:
                        print ("SUCC!\t%s => %s\t\t%d(%.5f) => %d(%.5f) %d(%.5f)" % \
                               (k, uid, y, old_prob, y, pr[y], p, pr[p]))
                        return True
                new_prob_idx = torch.argmin(new_prob[:, y])
                if new_prob[new_prob_idx][y] < old_prob:
                    x = new_x[new_prob_idx]
                    uids[new_uid_cand[new_prob_idx]] = uids.pop(k)
                    n_stop = 0
                    print("acc\t%s => %s\t\t%d(%.5f) => %d(%.5f)" % \
                          (k, new_uid_cand[new_prob_idx], y, old_prob, y, new_prob[new_prob_idx][y]))
                    old_prob = new_prob[new_prob_idx][y]
                else:
                    n_stop += 1
                    print("rej\t%s" % k)
        print("FAIL!")
        return False


    def _attack_ins(self,x_token, x, y, poses, rand_d, n_candidate=100, n_iter=20):
        self.insM.initInsertDict(poses)
        iter = 0
        n_stop = 0
        batch = get_batched_data([x], [y], self.txt2idx)
        old_prob = self.cl.prob(batch['x'])[0]

        if torch.argmax(old_prob) != y:
            print("torch.argmax(old_prob)=,",torch.argmax(old_prob))
            print("y=,",y)
            print("SUCC! Original mistake.")
            return True
        old_prob = old_prob[y]

        while iter < n_iter:
            iter += 1
            n_could_del = self.insM.insertDict["count"]
            n_candidate_del = n_could_del
            n_candidate_ins = n_candidate - n_candidate_del
            assert n_candidate_del >= 0 and n_candidate_ins >= 0
            new_x_del, new_insertDict_del = self.insM.remove(x, n_candidate_del)
            new_x_add, new_insertDict_add = self.insM.insert4(x_token,x, rand_d, n_candidate_ins)          
           
            new_x = new_x_del + new_x_add
            new_insertDict = new_insertDict_del + new_insertDict_add
            if new_x == []:
                n_stop += 1
                continue
            batch = get_batched_data(new_x, [y]*len(new_x), self.txt2idx) 
            new_prob = self.cl.prob(batch['x'])
            new_pred = torch.argmax(new_prob, dim=-1)
            
            for insD, p, pr in zip(new_insertDict, new_pred, new_prob):
                if p != y:
                    print("SUCC!\tinsert_n %d => %d\t\t%d(%.5f) => %d(%.5f) %d(%.5f)" % \
                          (self.insM.insertDict["count"], insD["count"],
                           y, old_prob, y, pr[y], p, pr[p]))
                    return True

            new_prob_idx = torch.argmin(new_prob[:, y])
            if new_prob[new_prob_idx][y] < old_prob:
                print("acc\tinsert_n %d => %d\t\t%d(%.5f) => %d(%.5f)" % \
                      (self.insM.insertDict["count"], new_insertDict[new_prob_idx]["count"],
                       y, old_prob, y, new_prob[new_prob_idx][y]))
                self.insM.insertDict = new_insertDict[new_prob_idx]
                n_stop = 0
                old_prob = new_prob[new_prob_idx][y]
            else:
                n_stop += 1
                print("rej\t%s" % "")
            if n_stop >= len(new_x):
                iter = n_iter
                break
        print("FAIL!")
        return False
    
    def attack_all(self, n_candidate=100, n_iter=20, dump_samples_path=None):
        n_succ = 0
        total_time = 0
 
        st_time = time.time()
        with open('/home/mabaoying/CARROT-main/preprocess-Author_data_txt/vocab_cnt_test_author.json', 'r') as VOCA:
            vocab_cnt_test = json.load(VOCA)
        loss_df = pd.DataFrame(columns=['test_num', 'n_succ', 'succ rate', 'call times', 'Aver call times'])
        # print("Test set size:", self.d.test.get_size())

        for i in range(self.d.test.get_size()):
            b = self.d.test.next_batch(1)
            print("\t%d/%d\tID = %d\tY = %d" % (i+1, self.d.test.get_size(), b['id'][0], b['y'][0]))
            start_time = time.time()

            # Generate rand_d for ins attack
            rand_d = []
            for _ in range(30):
                rand = self.d.test.next_batch(1)
                rand_d.append(rand)

            tag= self.attack(b['raw'][0],b['x'][0], b['y'][0], self.syms['te'][b['id'][0]], self.inss['stmt_te'][b['id'][0]], rand_d, vocab_cnt_test,n_candidate, n_iter)
            # print("x=================",x)
            if tag:
                n_succ += 1
                total_time += time.time() - start_time
            if n_succ <= 0:
                print("\tCurr succ rate = %.3f, Avg time cost = NaN sec" % (n_succ/(i+1)), flush=True)
            else:
                print("\tCurr succ rate = %.3f, Avg time cost = %.1f sec, Avg call times = %d" % \
                    (n_succ/(i+1), total_time/n_succ, (CodeBERTClassifier.counter)), flush=True)

        avg_counter = CodeBERTClassifier.counter / n_succ if n_succ > 0 else 0
        print("[Task Done] Time Cost: %.1f sec Succ Rate: %.3f Aver call times: %.3f" % \
              (time.time()-st_time, n_succ/self.d.test.get_size(), avg_counter))
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--data', type=str, default="../data_author/author.pkl.gz")
    parser.add_argument('--model_dir', type=str, default="../model/codebert/best_model")
    parser.add_argument('--bs', type=int, default=16)

    opt = parser.parse_args()
    if int(opt.gpu) < 0:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{opt.gpu}")  # <-- 改这一行
    print("Using device:", device)
    n_class = 66

    batch_size = opt.bs
    rand_seed = 1726
    
    torch.manual_seed(rand_seed)
    random.seed(rand_seed)
    numpy.random.seed(rand_seed)

    poj = Author66(path=opt.data)
    training_set = poj.train
    valid_set = poj.dev
    test_set = poj.test

    # import transformers after gpu selection
    from codebert import CodeBERTClassifier

    with gzip.open('../data_author/author_uid.pkl.gz', "rb") as f:
        symtab = pickle.load(f)
    with gzip.open('../data_author/author_inspos.pkl.gz', "rb") as f:
        instab = pickle.load(f)
        
    classifier = CodeBERTClassifier(model_path=opt.model_dir,
                                    num_labels=n_class,
                                    device=device).to(device)
    classifier.eval()

    attacker = CombinedAttacker(poj, symtab, instab, classifier)
    attacker.attack_all(15, 30)
