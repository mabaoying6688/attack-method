# -*- coding: utf-8 -*-

from dataset import Author66
from lscnn import LSCNNClassifier
from modifier import TokenModifier, InsModifier
from modifier import get_batched_data, gettensor
import numpy as np
import numpy
import random
import torch
import torch.nn as nn
import argparse
import pickle, gzip
import os, sys, time
import pandas as pd
from preprocesslstm import pattern 
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
        self.inss = instab

    def attack(self,x, y, uids, poses, rand_d=None, vocab_cnt_test=None,n_candidate=100, n_iter=20):
        # First, perform ins attack
        x_token=x
        ins_success, ins_adv_x, ins_adv_y = self._attack_ins(x_token,x, y, poses, rand_d, n_candidate, n_iter)
        if ins_success:
            return True, ins_adv_x, ins_adv_y
        token_success,ins_adv_x, ins_adv_y= self._attack_token(x, y, uids, vocab_cnt_test, n_candidate, n_iter)
        if token_success:
             return True, ins_adv_x, ins_adv_y

        return False,x,2
    
    def _attack_ins(self,x_token, x_raw, y, poses,rand_d, n_candidate=100, n_iter=20):
        
        self.insM.initInsertDict(poses)

        iter = 0
        n_stop = 0
        # print(self.cl.device)
        batch = get_batched_data([x_raw], [y], self.txt2idx, 
                                 self.max_stmt_cnt, self.max_stmt_len, self.cl.vocab_size)
        inputs, lens, labels = gettensor(batch, self.cl.device)
        old_prob = self.cl.prob(inputs, lens)[0]
        # print("torch.argmax(old_prob)",torch.argmax(old_prob))
        # print("y",y)
        if torch.argmax(old_prob) != y:
            print ("SUCC! Original mistake.")
            return True,x_raw,0
        old_prob = old_prob[y]
        
        while iter < n_iter:
            iter += 1

            # get insertion candidates
            n_could_del = self.insM.insertDict["count"]
            n_candidate_del = n_could_del
            n_candidate_ins = n_candidate - n_candidate_del
            assert n_candidate_del >= 0 and n_candidate_ins >= 0
            new_x_raw_del, new_insertDict_del = self.insM.remove(x_raw, n_candidate_del)
            new_x_raw_add, new_insertDict_add = self.insM.insert4(x_token,x_raw, rand_d, n_candidate_ins)
            new_x_raw = new_x_raw_del + new_x_raw_add
            new_insertDict = new_insertDict_del + new_insertDict_add
            if new_x_raw == []: # no valid candidates
                n_stop += 1
                continue

            # find if there is any candidate successful wrong classfied
            batch = get_batched_data(new_x_raw, [y]*len(new_x_raw), self.txt2idx, 
                                     self.max_stmt_cnt, self.max_stmt_len, self.cl.vocab_size)
            inputs, lens, labels = gettensor(batch, self.cl.device)
            new_prob = self.cl.prob(inputs, lens)
            new_pred = torch.argmax(new_prob, dim=-1)
            for insD, p, pr,x_ in zip(new_insertDict, new_pred, new_prob, new_x_raw):
                if p != y:
                    print ("SUCC!\tinsert_n %d => %d\t\t%d(%.5f) => %d(%.5f) %d(%.5f)" % \
                            (self.insM.insertDict["count"], insD["count"], 
                                y, old_prob, y, pr[y], p, pr[p]))
                    return True,x_,1

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
            if n_stop >= len(new_x_raw):    # len(new_x) could be smaller than n_candidate
                iter = n_iter
                break
        print ("FAIL!")
        return False,x_raw,2
    
    def attack(self, x_raw, y, uids, vocab_cnt_test, n_candidate=100, n_iter=20):        
        iter = 0
        n_stop = 0
        batch = get_batched_data([x_raw], [y], self.txt2idx, self.max_stmt_cnt, self.max_stmt_len, self.cl.vocab_size)
        inputs, lens, labels = gettensor(batch, self.cl.device)
        old_prob = self.cl.prob(inputs, lens)[0]
        if torch.argmax(old_prob) != y:
            print ("SUCC! Original mistake.")
            return True, x_raw,0
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
                new_x_raw, new_x_uid = self.tokenM.rename_uid(x_raw, y,vocab_cnt_test, k, n_candidate)
                if new_x_raw is None:
                    n_stop += 1
                    print ("skip unk\t%s" % k)
                    continue
                batch = get_batched_data(new_x_raw, [y]*len(new_x_raw), self.txt2idx, 
                                         self.max_stmt_cnt, self.max_stmt_len, self.cl.vocab_size)
                inputs, lens, labels = gettensor(batch, self.cl.device)
                new_prob = self.cl.prob(inputs, lens)
                new_pred = torch.argmax(new_prob, dim=-1)
                for uid, p, pr, x_ in zip(new_x_uid, new_pred, new_prob, new_x_raw):
                    if p != y:
                        print ("SUCC!\t%s => %s\t\t%d(%.5f) => %d(%.5f) %d(%.5f)" % \
                               (k, self.idx2txt[uid], y, old_prob, y, pr[y], p, pr[p]))
                        return True, x_,1
                new_prob_idx = torch.argmin(new_prob[:, y])
                if new_prob[new_prob_idx][y] < old_prob:
                    x_raw = new_x_raw[new_prob_idx]
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
        return False, x_raw, 2
    
    def attack_all(self, n_candidate=100, n_iter=20, res_save=None,adv_sample_size=200):
        
        n_succ = 0
        total_time = 0
        st_time = time.time()
        adv_xs, adv_labels, adv_ids,adv_ids = [], [], [], []
        fail_pred_xs, fail_pred_labels, fail_pred_ids = [], [], []   
        with open('/home/mabaoying/CARROT-main/preprocess-Author_data_txt/vocab_cnt_test_author.json', 'r') as VOCA:  
             vocab_cnt_test = json.load(VOCA)  
        for i in range(self.d.train.get_size()):
            b = self.d.train.next_batch(1)
            print ("\t%d/%d\tID = %d\tY = %d" % (i+1, self.d.train.get_size(), b['id'][0], b['y'][0]))
            print("b['raw'][0]====================",b['raw'][0])

            start_time = time.time()
            # Generate rand_d for ins attack
            rand_d = []
            for _ in range(30):
                rand = self.d.train.next_batch(1)
                rand_d.append(rand) 
            tag, x, typ= self.attack(b['raw'][0], b['y'][0], self.syms['tr'][b['id'][0]], self.inss['stmt_tr'][b['id'][0]], rand_d, vocab_cnt_test,n_candidate, n_iter)                           
            if tag:
                n_succ += 1
                total_time += time.time() - start_time
                # fail_pred_x1s.append(x)
                # fail_pred_labels.append(int(b['y'][0]))
                fail_pred_xs.append(x)
                fail_pred_labels.append(int(b['y'][0]))
                fail_pred_ids.append(b['id'][0])
            if typ==1:
                # adv_x1s.append(x)
                # adv_labels.append(int(b['y'][0])) 
                adv_xs.append(x)
                adv_labels.append(int(b['y'][0]))
                adv_ids.append(b['id'][0])
                total_time += time.time() - start_time
            if n_succ <= 0:
                print ("\tCurr succ rate = %.3f, Avg time cost = NaN sec" \
                       % (n_succ/(i+1)), flush=True)
            else:
                # print ("\tCurr succ rate = %.3f, Avg time cost = %.1f sec" \
                #        % (n_succ/(i+1), total_time/n_succ), flush=True)
                print ("\tCurr succ rate = %.3f, Avg time cost = %.1f sec, Call times = %d " \
                       % (n_succ/(i+1), total_time/n_succ, LSCNNClassifier.counter), flush=True)
        if res_save is not None:
            print ("Adversarial Sample Number: %d (Out of %d False Predicted Sample)" % (len(adv_xs), len(fail_pred_xs)))
            with gzip.open(res_save, "wb") as f:
                #unpadding_adv_xs = [remove_tail_padding(adv_x, 0) for adv_x in adv_xs]
                pickle.dump({"fail_pred_x": fail_pred_xs, 
                             "fail_pred_label": fail_pred_labels,
                             "fail_pred_id": fail_pred_ids,
                             "adv_x": adv_xs, 
                             #"adv_raw": self.d.idxs2raw(unpadding_adv_xs, [len(x) for x in unpadding_adv_xs]),
                             #"adv_raw": self.d.idxs2raw(adv_xs, [len(x) for x in adv_xs]),
                             "adv_label": adv_labels,
                             "adv_id": adv_ids}, f)            
        print("[Task Done] Time Cost: %.1f sec Succ Rate: %.3f Aver Call times: %.3f" % (time.time()-st_time, n_succ/self.d.train.get_size(),LSCNNClassifier.counter/n_succ))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default="1")
    parser.add_argument('--data', type=str, default="../data_author/author.pkl.gz")
    parser.add_argument('--model_path', type=str, default="../model/author-lscnn/best_model.pt")
    parser.add_argument('--bs', type=int, default=16)

    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    if int(opt.gpu) < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    n_class = 66
    vocab_size = 2300
    batch_size = opt.bs
    rand_seed = 1726
##
    embed_width = 300
    n_conv = 4096
    conv_size = 7
    lstm_size = 300
    n_lstm = 1
    max_stmt_cnt = 40
    max_stmt_len = 20
    brnn = True
##
    torch.manual_seed(rand_seed)
    random.seed(rand_seed)
    numpy.random.seed(rand_seed)

    poj = Author66(path=opt.data,
                max_stmt_len=max_stmt_len,
                max_stmt_cnt=max_stmt_cnt,
                vocab_size=vocab_size)
    training_set = poj.train
    valid_set = poj.dev
    test_set = poj.test

    with gzip.open('../data_author/author_uid.pkl.gz', "rb") as f:
        symtab = pickle.load(f)
    with gzip.open('../data_author/author_inspos.pkl.gz', "rb") as f:
        instab = pickle.load(f)
        

    classifier = LSCNNClassifier(n_class, vocab_size, embed_width,
                                n_conv, conv_size, lstm_size, n_lstm,
                                brnn, device).to(device)

    classifier.load_state_dict(torch.load(opt.model_path, map_location=device))  
    classifier.device = device

    attacker = CombinedAttacker(poj, symtab, instab, classifier)
    attacker.attack_all(20, 30, res_save="../author_lscnn.advsamples.pkl.gz", adv_sample_size=500) 
