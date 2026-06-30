# -*- coding: utf-8 -*-

from dataset import OJ104, remove_tail_padding
from lstm_classifier import LSTMClassifier, LSTMEncoder
from modifier import TokenModifier, InsModifier
import pandas as pd
import random
import json
import torch
import argparse
import pickle, gzip
import os, sys, time, copy
#AdversarialTrainingAttacker_MHM
import numpy as np

class CombinedAttacker(object):
    
    def __init__(self, dataset, symtab, instab, classifier):
        self.tokenM = TokenModifier(classifier=classifier,
                                    loss=torch.nn.CrossEntropyLoss(),
                                    uids=symtab['all'],
                                    txt2idx=dataset.get_txt2idx(),
                                    idx2txt=dataset.get_idx2txt())
        self.insM = InsModifier(classifier=classifier,
                                txt2idx=dataset.get_txt2idx(),
                                poses=None)  # wait to init when attack
        self.cl = classifier
        self.d = dataset
        self.syms = symtab
        self.inss = instab
    
    def attack(self,x_token,x, y, uids, poses, rand_d=None, vocab_cnt_test=None,n_candidate=100, n_iter=20):
        ins_success, ins_adv_x, ins_adv_y = self._attack_ins(x_token,x, y, poses, rand_d, n_candidate, n_iter)
        if ins_success:
            return True, ins_adv_x, ins_adv_y
        token_success, token_adv_x, token_adv_y = self._attack_token(x, y, uids, vocab_cnt_test, n_candidate, n_iter)
        if token_success:
            return True, token_adv_x, token_adv_y
        return False, x, y
    
    def _attack_token(self, x, y, uids, vocab_cnt_test, n_candidate=100, n_iter=20):
        iter = 0
        n_stop = 0
        ori_x = copy.deepcopy(x)
        old_prob = self.cl.prob(torch.tensor(x, dtype=torch.long).cuda().permute([1, 0]))[0]

        if torch.argmax(old_prob) != y[0]:
            print("SUCC! Original mistake.")
            return True, x, 0

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
                new_x, new_uid_cand = self.tokenM.rename_uid(uids, x, y, uids[k], vocab_cnt_test, k, n_candidate)
                if new_x is None:
                    n_stop += 1
                    print ("skip unk\t%s" % k)
                    continue
                new_prob = self.cl.prob(torch.tensor(new_x, dtype=torch.long).cuda().permute([1, 0]))
                new_pred = torch.argmax(new_prob, dim=1)
                for uid, p, pr, _x in zip(new_uid_cand, new_pred, new_prob, new_x):
                    if p != y[0]:
                        print("SUCC!\t%s => %s\t\t%d(%.5f) => %d(%.5f) %d(%.5f)" % \
                              (k, self.d.idx2vocab(uid), y[0], old_prob, y[0], pr[y[0]], p, pr[p]))
                        return True, [_x], 1
                new_prob_idx = torch.argmin(new_prob[:, y[0]])
                if new_prob[new_prob_idx][y[0]] < old_prob:
                    x = [new_x[new_prob_idx]]
                    uids[self.d.idx2vocab(int(new_uid_cand[new_prob_idx]))] = uids.pop(k)
                    n_stop = 0
                    print("acc\t%s => %s\t\t%d(%.5f) => %d(%.5f)" % \
                          (k, self.d.idx2vocab(int(new_uid_cand[new_prob_idx])),
                          y[0], old_prob, y[0], new_prob[new_prob_idx][y[0]]))
                    old_prob = new_prob[new_prob_idx][y[0]]
                else:
                    n_stop += 1
                    print("rej\t%s" % k)
        print("FAIL!")
        return False, x, 2

    def _attack_ins(self,x_token, x, y, poses, rand_d, n_candidate=100, n_iter=20):
        self.insM.initInsertDict(poses)
        ori_x = copy.deepcopy(x)
        iter = 0
        n_stop = 0
        old_prob = self.cl.prob(torch.tensor(x, dtype=torch.long).cuda().permute([1, 0]))[0]

        if torch.argmax(old_prob) != y[0]:
            print("SUCC! Original mistake.")
            return True, x, 0

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
            if new_x == []:
                n_stop += 1
                continue

            feed_new_x = [_x[:self.cl.max_len] for _x in new_x]
            feed_tensor = torch.tensor(feed_new_x, dtype=torch.long)
            new_prob = self.cl.prob(feed_tensor.cuda().permute([1, 0]))
            new_pred = torch.argmax(new_prob, dim=1)
            for insD, p, pr, _x in zip(new_insertDict, new_pred, new_prob, new_x):
                if p != y[0]:
                    print("SUCC!\tinsert_n %d => %d\t\t%d(%.5f) => %d(%.5f) %d(%.5f)" % \
                          (self.insM.insertDict["count"], insD["count"],
                           y[0], old_prob, y[0], pr[y[0]], p, pr[p]))
                    return True, [_x], 1

            new_prob_idx = torch.argmin(new_prob[:, y[0]])
            if new_prob[new_prob_idx][y[0]] < old_prob:
                print("acc\tinsert_n %d => %d\t\t%d(%.5f) => %d(%.5f)" % \
                      (self.insM.insertDict["count"], new_insertDict[new_prob_idx]["count"],
                       y[0], old_prob, y[0], new_prob[new_prob_idx][y[0]]))
                self.insM.insertDict = new_insertDict[new_prob_idx]
                n_stop = 0
                old_prob = new_prob[new_prob_idx][y[0]]
            else:
                n_stop += 1
                print("rej\t%s" % "")
            if n_stop >= len(new_x):
                iter = n_iter
                break
        print("FAIL!")
        return False, ori_x, 2
    
    def attack_all(self, n_candidate=100, n_iter=20, res_save=None, adv_sample_size=None):
        sample_dict = {"x": [], "y": [], "adv_x": [], "adv_y": []}
        n_succ, n_total = 0, 0
        total_time = 0
        adv_xs, adv_labels, adv_ids,adv_ids = [], [], [], []
        fail_pred_xs, fail_pred_labels, fail_pred_ids = [], [], []
        st_time = time.time()
        with open('../preprocesslstm/vocab_cnt_test.json', 'r') as VOCA:
            vocab_cnt_test = json.load(VOCA)
        loss_df = pd.DataFrame(columns=['test_num', 'n_succ', 'succ rate', 'call times', 'Aver call times'])
        for i in range(self.d.train.get_size()):
            if len(adv_xs) >= adv_sample_size:
                break
            b = self.d.train.next_batch(1)
            print ("\t%d/%d\tID = %d\tY = %d" % (i+1, self.d.train.get_size(), b['id'][0], b['y'][0]))
            sample_dict["x"].append(b['x'][0])
            sample_dict["y"].append(b['y'][0])
            # print("\t%d/%d\tID = %d\tY = %d" % (i+1, self.d.test.get_size(), b['id'][0], b['y'][0]))
            start_time = time.time()
            # Generate rand_d for ins attack
            rand_d = []
            for _ in range(30):
                rand = self.d.test.next_batch(1)
                rand_d.append(rand)
            # Perform combined attack
            tag, x, typ = self.attack(b['raw'],b['x'], b['y'], self.syms['tr'][b['id'][0]], self.inss['stmt_tr'][b['id'][0]], rand_d, vocab_cnt_test,n_candidate, n_iter)
            x = x[0]
            print("x=======================",x)
            if tag:
                n_succ += 1
                total_time += time.time() - start_time
                fail_pred_xs.append(x)
                fail_pred_labels.append(int(b['y'][0]))
                fail_pred_ids.append(b['id'][0])
            if typ == 1:
                adv_xs.append(x)
                adv_labels.append(int(b['y'][0]))
                adv_ids.append(b['id'][0])
            if n_succ <= 0:
                print ("\tCurr succ rate = %.3f, Avg time cost = NaN sec" \
                       % (n_succ/(i+1)), flush=True)
            else:
                print ("\tCurr succ rate = %.3f, Avg time cost = %.1f sec" \
                       % (n_succ/(i+1), total_time/n_succ), flush=True)
            n_total += 1
        if res_save is not None:
            print ("Adversarial Sample Number: %d (Out of %d False Predicted Sample)" % (len(adv_xs), len(fail_pred_xs)))
            with gzip.open(res_save, "wb") as f:
                unpadding_adv_xs = [remove_tail_padding(adv_x, 0) for adv_x in adv_xs]
                pickle.dump({"fail_pred_x": fail_pred_xs, 
                             "fail_pred_label": fail_pred_labels,
                             "fail_pred_id": fail_pred_ids,
                             "adv_x": adv_xs, 
                             "adv_raw": self.d.idxs2raw(unpadding_adv_xs, [len(x) for x in unpadding_adv_xs]),
                             "adv_label": adv_labels,
                             "adv_id": adv_ids}, f)
        print("[Task Done] Time Cost: %.1f sec Succ Rate: %.3f" % (time.time()-st_time, n_succ/n_total))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', default="1")
    parser.add_argument('-attn', action='store_true')
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    device = torch.device("cuda")
    
    vocab_size = 5000
    embedding_size = 512
    hidden_size = 600
    n_layers = 2
    num_classes = 104
    max_len = 600

    poj = OJ104(path="../data/oj.pkl.gz",
                max_len=max_len,
                vocab_size=vocab_size)
    training_set = poj.train
    valid_set = poj.dev
    test_set = poj.test
    with gzip.open('../data/oj_uid.pkl.gz', "rb") as f:
        symtab = pickle.load(f)
    with gzip.open('../data/oj_inspos.pkl.gz', "rb") as f:
        instab = pickle.load(f)
    enc = LSTMEncoder(embedding_size, hidden_size, n_layers)
    classifier = LSTMClassifier(vocab_size, embedding_size, enc,
                                hidden_size, num_classes, max_len, attn=opt.attn).cuda()
    classifier.load_state_dict(torch.load('../model/PATH45.pt'))

    attacker = CombinedAttacker(poj, symtab, instab, classifier)
    attacker.attack_all(40, 20, res_save="./lstm_oj_atk.advsamples.pkl.gz", adv_sample_size=5000) 

    