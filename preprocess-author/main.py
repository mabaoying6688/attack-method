# -*- coding: utf-8 -*-
import json
import shutil
import os, sys
import pickle, gzip
import mytoken as tk
import build_dataset as bd
from pattern import StmtInsPos, DeclInsPos
from tqdm import tqdm

def dataset(dir = '../preprocess-Author_data/tmp', tgt = '../data/author.pkl.gz',
            symtab = '../data/author_uid.pkl.gz', 
            inspos_file = '../data/author_inspos.pkl.gz',
            done_file = 'dataset.done'):
    
    if tk.unzip():
        d = tk.tokenize()
        # for i in range(min(3, len(d["raw"]))):
        #     print(f"\n=================================== 样本 {i} =====")
        #     print("Label:=====================================", d["labels"][i])
        #     print("Tokens=====================================:", d["raw"][i])
        #     print("UIDs:=======================================", d["uids"][i])
        if d is not None:
            train, test = bd.split(d)
            idx2txt, txt2idx,_ = bd.build_vocab(train['raw'])
            idx2txt_test, txt2idx_test,vocab_cnt_test = bd.build_vocab(test['raw']) 
            with open('vocab_cnt_test_author.json', 'w') as VOC:  
                json.dump(vocab_cnt_test, VOC) 
            train_tokens = bd.text2index(train['raw'], txt2idx)
            test_tokens = bd.text2index(test['raw'], txt2idx)
            uids = []
            for _uids in train["uids"]:
                for _uid in _uids.keys():
                    if _uid not in uids:
                        uids.append(_uid)
            if not os.path.isfile(os.path.join(dir, done_file)):
                data = {"raw_tr": train["raw"], "y_tr": train["labels"],
                        "x_tr": train_tokens,
                        "raw_te": test["raw"], "y_te": test["labels"],
                        "x_te": test_tokens,
                        "idx2txt": idx2txt, "txt2idx": txt2idx}
                uid = {"tr": train["uids"], "te": test["uids"], "all": uids}
                with gzip.open(tgt, "wb") as f:
                    pickle.dump(data, f)
                with gzip.open(symtab, "wb") as f:
                    pickle.dump(uid, f)
                with open(os.path.join(dir, done_file), "wb") as f:
                    pass    
                stmt_poses_tr = [StmtInsPos(tr, strict=False) for tr in tqdm(train['raw'])]
                stmt_poses_te = [StmtInsPos(te, strict=False) for te in tqdm(test['raw'])]
                decl_poses_tr = [DeclInsPos(tr) for tr in tqdm(train['raw'])]
                decl_poses_te = [DeclInsPos(te) for te in tqdm(test['raw'])]
                inspos = {"stmt_tr": stmt_poses_tr, "stmt_te": stmt_poses_te, 
                          "decl_tr": decl_poses_tr, "decl_te": decl_poses_te}
                
                with gzip.open(inspos_file, "wb") as f:
                    pickle.dump(inspos, f)       
    #shutil.rmtree(dir)

if __name__ == "__main__":
    
    dataset()
    