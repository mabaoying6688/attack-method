# -*- coding: utf-8 -*-

import keyword
import json
import pandas as pd
from dataset import OJ104
from modifier import TokenModifier, InsModifier
from modifier import get_batched_data
from sklearn.metrics import precision_recall_fscore_support
import numpy
import numpy as np
import random
import torch
import torch.nn as nn
import argparse
import pickle, gzip
import os, sys, time
import transformers
transformers.logging.set_verbosity_error()
import pattern


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
                                #后添加
                                idx2txt=dataset.get_idx2txt(),
                                poses=None)  # wait to init when attack
        self.cl = classifier
        self.d = dataset
        self.syms = symtab
        self.inss = instab

    def attack(self,x_raw,x_raw2,x, x2, y, uids,poses,rand_d,vocab_cnt_test=None,n_candidate=100, n_iter=20):
        x_token=x_raw[0]
        ins_success,prep = self._attack_ins(x_token,x_raw[0],x_raw2[0], y, poses, rand_d, n_candidate, n_iter)
        if ins_success:
            return True,prep
        
        token_success,x,x2,typ,pred= self._attack_token(x_raw[0],x_raw2[0], y, uids, vocab_cnt_test, n_candidate, n_iter)
        if token_success:
            return True,x,x2,typ,pred           
        return False,x,x2,typ, y
    
    def _attack_token(self, x, x2, y, uids, vocab_cnt_test, n_candidate=100, n_iter=20):
        iter = 0
        n_stop = 0
        batch = get_batched_data([x], [x2], [y], self.txt2idx)
        old_prob = self.cl.prob(batch['x1'], batch['x2'])[0]  # shape: [num_labels]
        if torch.argmax(old_prob) != y[0]:
            
            print ("SUCC! Original mistake.")
            return True, x,x2,0,[torch.argmax(old_prob).cpu().numpy()]
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
                new_x, new_uid_cand = self.tokenM.rename_uid(uids, x, x2, y, uids[k], vocab_cnt_test, k, n_candidate)
                if new_x is None:
                    n_stop += 1
                    print ("skip unk\t%s" % k)
                    continue
                batch = get_batched_data(new_x, [x2]*len(new_x), [y]*len(new_x), self.txt2idx)
                new_prob = self.cl.prob(batch['x1'], batch['x2'])  # shape: [num_candidates, num_labels]
                new_pred = torch.argmax(new_prob, dim=1)
                
                for uid, p, pr, _x in zip(new_uid_cand, new_pred, new_prob, new_x):
                    # print("p =",p)
                    # print("y =",y)
                    if p != y[0]:
                        print ("SUCC!\t%s => %s\t\t%d(%.5f) => %d(%.5f) %d(%.5f)" %
                            (k, uid, y, old_prob, y, pr[y], p, pr[p]))
                        return True, [_x],x2,1,[p.cpu().numpy()]

                new_prob_idx = torch.argmin(new_prob[:, y])
                if new_prob[new_prob_idx][y] < old_prob:
                    x= new_x[new_prob_idx]
                    uids[new_uid_cand[new_prob_idx]] = uids.pop(k)
                    n_stop = 0
                    print("acc\t%s => %s\t\t%d(%.5f) => %d(%.5f)" %
                        (k, new_uid_cand[new_prob_idx], y, old_prob, y, new_prob[new_prob_idx][y]))
                    old_prob = new_prob[new_prob_idx][y]
                else:
                    n_stop += 1
                    print("rej\t%s" % k)

        print("FAIL!")
        return False, x, x2, 2, y

    def _attack_ins(self,x_token, x, x2, y, poses, rand_d, n_candidate=100, n_iter=20):
        self.insM.initInsertDict(poses)
        iter = 0
        n_stop = 0
        batch = get_batched_data([x], [x2], [y], self.txt2idx)
        old_prob = self.cl.prob(batch['x1'], batch['x2'])[0]
        if torch.argmax(old_prob) != y[0]:
            # print("torch.argmax(old_prob)=，",torch.argmax(old_prob))
            # print("y=，",y)
            print("SUCC! Original mistake.")
            return True, x, x2, 0,[torch.argmax(old_prob).cpu().numpy()]
        old_prob = old_prob[y]

        while iter < n_iter:
            iter += 1
            n_could_del = self.insM.insertDict["count"]
            n_candidate_del = n_could_del
            n_candidate_ins = n_candidate - n_candidate_del
            assert n_candidate_del >= 0 and n_candidate_ins >= 0
            new_x_del, new_insertDict_del = self.insM.remove(x, n_candidate_del)
            new_x_add, new_insertDict_add = self.insM.insert4(x_token,x, x2, rand_d, n_candidate_ins)          
            new_x = new_x_del + new_x_add
            new_insertDict = new_insertDict_del + new_insertDict_add
            if new_x == []:
                n_stop += 1
                continue
            batch = get_batched_data(new_x,[x2]*len(new_x),  [y]*len(new_x), self.txt2idx) 
            new_prob = self.cl.prob(batch['x1'], batch['x2'])
            new_pred = torch.argmax(new_prob, dim=-1)
            
            for insD, p, pr,_x in zip(new_insertDict, new_pred, new_prob, new_x):
                if p != y[0]:
                    print("SUCC!\tinsert_n %d => %d\t\t%d(%.5f) => %d(%.5f) %d(%.5f)" % \
                          (self.insM.insertDict["count"], insD["count"],
                           y, old_prob, y, pr[y], p, pr[p]))
                    return True, [_x], x2, 1,[p.cpu().numpy()]

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
        return False, x, x2, 2, y
    
    def attack_all(self, n_candidate=100, n_iter=20,res_save=None,adv_sample_size=2000):
        # sample_dict = {"x": [], "y": [], "adv_x": [], "adv_y": []}
        n_succ = 0
        total_time = 0
        trues = []
        preds = []
        x1s, x2s = [], []
        raw1s, raw2s = [], []
        adv_x1s, adv_x2s, adv_labels = [], [], []
        fail_pred_x1s, fail_pred_x2s, fail_pred_labels, fail_pred_ids  = [], [], [], []
        st_time = time.time()
        with open('/home/mabaoying/CARROT-main/vocab_cnt_test.json', 'r') as VOCA:
            vocab_cnt_test = json.load(VOCA)
        loss_df = pd.DataFrame(columns=['test_num', 'n_succ', 'succ rate', 'call times', 'Aver call times'])

        for i in range(self.d.train.get_size()):
            if len(adv_x1s) >= adv_sample_size:
                break
            b = self.d.train.next_batch(1)

            print ("\t%d/%d\tID = (%d, %d)\tY = %d" %
                   (i+1, self.d.train.get_size(), b['id1'][0], b['id2'][0], b['y'][0]))
            start_time = time.time()
            rand_d = []
            for _ in range(30):
                rand = self.d.test.next_batch(1)
                rand_d.append(rand)
            # Perform combined attack，，tag, adv_x, adv_y
            # print("Type of b['raw'][0]:", type(b['raw'][0]))
            # print("Sample of b['raw'][0]:", b['raw'][0][:10])  # 打印前10个元素          
            tag,x,x2,typ,pred = self.attack(b['raw1'],b['raw2'],b['x1'], b['x2'], b['y'], self.syms['tr'][b['id1'][0]], self.inss['stmt_tr'][b['id1'][0]],
                                           rand_d,vocab_cnt_test,n_candidate, n_iter)            
            x, x2 = x[0], x2[0]

            if tag:
                n_succ += 1
                total_time += time.time() - start_time
                fail_pred_x1s.append(x)
                fail_pred_x2s.append(x2)
                fail_pred_labels.append(int(b['y'][0]))

            preds.append(int(pred[0]))
            trues.append(int(b['y'][0]))
            if typ == 1:
                adv_x1s.append(x)
                adv_x2s.append(x2)
                adv_labels.append(int(b['y'][0]))  
            if n_succ <= 0:
                print ("\tCurr succ rate = %.3f, Avg time cost = NaN sec" \
                       % (n_succ/(i+1)), flush=True)
            else:
                print("\tCurr succ rate = %.3f, Avg time cost = %.1f sec, Avg call times = %d" % \
                      (n_succ/(i+1), total_time/n_succ, (CodeBERTClassifier.counter)), flush=True)
                new_data = pd.DataFrame({
                    'test_num': [i + 1],
                    'n_succ': [n_succ],
                    'succ rate': [n_succ/(i+1)],
                    'call times': [(CodeBERTClassifier.counter)],
                    'Aver call times': [(CodeBERTClassifier.counter)/n_succ]
                })
                loss_df = pd.concat([loss_df, new_data], ignore_index=True)
                loss_df.to_csv('CodeBERT_oj-succ_values.csv', index=False)

            precision, recall, f1, _ = precision_recall_fscore_support(trues, preds, average='binary')
            print("\t(P, R, F1) = (%.3f, %.3f, %.3f)" % (precision, recall, f1))
        if res_save is not None:
            print ("Adversarial Sample Number: %d (Out of %d False Predicted Sample)" % (len(adv_x1s), len(fail_pred_x1s)))
            with gzip.open(res_save, "wb") as f:
                pickle.dump({"fail_pred_x1s": fail_pred_x1s, 
                             "fail_pred_x2s": fail_pred_x2s, 
                             "fail_pred_labels": fail_pred_labels,
                             "adv_x1s": adv_x1s,
                             "adv_x2s": adv_x2s,
                             "adv_labels": adv_labels}, f)

        print("[Task Done] Time Cost: %.1f sec Succ Rate: %.3f Aver call times: %.3f" % \
              (time.time()-st_time, n_succ/self.d.train.get_size(), (CodeBERTClassifier.counter)/n_succ))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)  
    parser.add_argument('--data', type=str, default="../data_clone/oj.pkl.gz")
    parser.add_argument('--model_dir', type=str, default="../model/codebert-clone/clone_new/15-1.pt")
    parser.add_argument('--bs', type=int, default=16)

    opt = parser.parse_args()
    torch.cuda.set_device(opt.gpu)
    if int(opt.gpu) < 0:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{opt.gpu}")  
    print("Using device:", device)
    n_class = 2

    batch_size = opt.bs
    rand_seed = 1726
    
    torch.manual_seed(rand_seed)
    random.seed(rand_seed)
    numpy.random.seed(rand_seed)

    poj = OJ104(path=opt.data)
    training_set = poj.train
    valid_set = poj.dev
    test_set = poj.test

    # import transformers after gpu selection
    from codebert import CodeBERTClassifier

    with gzip.open('../data_clone/oj_uid.pkl.gz', "rb") as f:
        symtab = pickle.load(f)
    with gzip.open('../data_clone/oj_inspos.pkl.gz', "rb") as f:
        instab = pickle.load(f)
        
    classifier = CodeBERTClassifier(model_path=opt.model_dir,
                                    num_labels=n_class,
                                    device=device).to(device)
    classifier.eval()

 
    attacker = CombinedAttacker(poj, symtab, instab, classifier)
    attacker.attack_all(10, 30, "../codebert_OJclone-atk.advsamples.pkl.gz", 5000)
