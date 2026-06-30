# -*- coding: utf-8 -*-
import shutil
import os, sys
import pickle, gzip
import mytoken as tk
import build_dataset as bd
from patternpy import StmtInsPos, DeclInsPos
from tqdm import tqdm
import json
          
def dataset(dir = './tmp', tgt = '../data/oj.pkl.gz',  
            symtab = '../data/oj_uid.pkl.gz',       
            inspos_file = '../data/oj_inspos.pkl.gz',    
            done_file = 'dataset.done'):     
    if tk.unzip():  
        d = tk.tokenize()   
        if d is not None:
            train, test = bd.split(d)
            print("Current Working Directory:", os.getcwd()) 
            output_txt_path = "../train_data.txt"
            with open(output_txt_path, "w", encoding="utf-8") as output_file:
                json.dump(d, output_file, ensure_ascii=False, indent=4)
            try:
                with open(output_txt_path, "w", encoding="utf-8") as output_file:
                    for item in train:
                        json.dump(item, output_file, ensure_ascii=False)
                        output_file.write("\n") 
            except Exception as e:
                print("An error occurred:", str(e))
            idx2txt, txt2idx,vocab_cnt_tr = bd.build_vocab(train['raw']) 
            idx2txt_test, txt2idx_test,vocab_cnt_test = bd.build_vocab(test['raw']) 
            with open('vocab_cnt_test.json', 'w') as VOC:  
                 json.dump(vocab_cnt_test, VOC)
            train_tokens = bd.text2index(train['raw'], txt2idx) 
            test_tokens = bd.text2index(test['raw'], txt2idx)
            uids = []
            for _uids in train["uids"]:
                for _uid in _uids.keys():
                    if _uid not in uids:
                        uids.append(_uid)
            counts = {}
            for item in train["uids"]:
              for key, value in item.items():
                 count = len(value)
                 if key in counts:
                    counts[key] += count
                 else:
                    counts[key] = count
            with open('my-train-uids-num.txt', 'w') as file:
               for key, value in counts.items():
                  file.write(f"{key}: {value}\n")

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
                stmt_poses_tr = [StmtInsPos(tr) for tr in tqdm(train['raw'])]
                stmt_poses_te = [StmtInsPos(te) for te in tqdm(test['raw'])]  
                decl_poses_tr = [DeclInsPos(tr) for tr in tqdm(train['raw'])]
                decl_poses_te = [DeclInsPos(te) for te in tqdm(test['raw'])]
                inspos = {"stmt_tr": stmt_poses_tr, "stmt_te": stmt_poses_te, 
                          "decl_tr": decl_poses_tr, "decl_te": decl_poses_te}
                with gzip.open(inspos_file, "wb") as f:
                    pickle.dump(inspos, f)   
if __name__ == "__main__":   
    dataset()