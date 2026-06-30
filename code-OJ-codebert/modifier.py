# -*- coding: utf-8 -*-

import re
import torch
import random
import copy
import numpy
from copy import deepcopy
import sys
import math
import numpy as np
from  preprocesslstm import patternpy
from pycparser import c_parser
parser = c_parser.CParser()

def raw2x(raws, txt2idx):
    "here we dont convert raw to ids in fact, but replace OoVs as '<unk>'"
    xs = []

    for raw in raws:
        xs.append([])
        for token in raw:
            if token in txt2idx.keys():
                xs[-1].append(token)
            else:
                xs[-1].append("<unk>")
    return xs

def get_batched_data(raws, ys, txt2idx, ids=None):
    xs = raw2x(raws, txt2idx)
    batch = {"x": [], "y": [], "raw": [], "id": [], "new_epoch": False}
    batch['x'] = xs # is still token list, but with certain '<unk>'s
    batch['x'] = [" ".join(x) for x in xs]
    batch['y'] = ys
    batch['id'] = ids
    batch['raw'] = deepcopy(raws)    
    return batch

class TokenModifier(object):   
    def __init__(self, classifier, loss, uids, txt2idx, idx2txt):        
        self.cl = classifier
        self.loss = loss
        # poj's vocab, not codebert's vocab
        self.txt2idx = txt2idx
        self.idx2txt = idx2txt
        self.__key_words__ = ["auto", "break", "case", "char", "const", "continue",
                             "default", "do", "double", "else", "enum", "extern",
                             "float", "for", "goto", "if", "inline", "int", "long",
                             "register", "restrict", "return", "short", "signed",
                             "sizeof", "static", "struct", "switch", "typedef",
                             "union", "unsigned", "void", "volatile", "while",
                             "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex",
                             "_Generic", "_Imaginary", "_Noreturn", "_Static_assert",
                             "_Thread_local", "__func__"]
        self.__ops__ = ["...", ">>=", "<<=", "+=", "-=", "*=", "/=", "%=", "&=", "^=", "|=",
                       ">>", "<<", "++", "--", "->", "&&", "||", "<=", ">=", "==", "!=", ";",
                       "{", "<%", "}", "%>", ",", ":", "=", "(", ")", "[", "<:", "]", ":>",
                       ".", "&", "!", "~", "-", "+", "*", "/", "%", "<", ">", "^", "|", "?"]
        self.__macros__ = ["NULL", "_IOFBF", "_IOLBF", "BUFSIZ", "EOF", "FOPEN_MAX", "TMP_MAX",  # <stdio.h> macro
                          "FILENAME_MAX", "L_tmpnam", "SEEK_CUR", "SEEK_END", "SEEK_SET",
                          "NULL", "EXIT_FAILURE", "EXIT_SUCCESS", "RAND_MAX", "MB_CUR_MAX"]     # <stdlib.h> macro
        self.__special_ids__ = ["main",  # main function
                               "stdio", "cstdio", "stdio.h",                                # <stdio.h> & <cstdio>
                               "size_t", "FILE", "fpos_t", "stdin", "stdout", "stderr",     # <stdio.h> types & streams
                               "remove", "rename", "tmpfile", "tmpnam", "fclose", "fflush", # <stdio.h> functions
                               "fopen", "freopen", "setbuf", "setvbuf", "fprintf", "fscanf",
                               "printf", "scanf", "snprintf", "sprintf", "sscanf", "vprintf",
                               "vscanf", "vsnprintf", "vsprintf", "vsscanf", "fgetc", "fgets",
                               "fputc", "getc", "getchar", "putc", "putchar", "puts", "ungetc",
                               "fread", "fwrite", "fgetpos", "fseek", "fsetpos", "ftell",
                               "rewind", "clearerr", "feof", "ferror", "perror", "getline"
                               "stdlib", "cstdlib", "stdlib.h",                             # <stdlib.h> & <cstdlib>
                               "size_t", "div_t", "ldiv_t", "lldiv_t",                      # <stdlib.h> types
                               "atof", "atoi", "atol", "atoll", "strtod", "strtof", "strtold",  # <stdlib.h> functions
                               "strtol", "strtoll", "strtoul", "strtoull", "rand", "srand",
                               "aligned_alloc", "calloc", "malloc", "realloc", "free", "abort",
                               "atexit", "exit", "at_quick_exit", "_Exit", "getenv",
                               "quick_exit", "system", "bsearch", "qsort", "abs", "labs",
                               "llabs", "div", "ldiv", "lldiv", "mblen", "mbtowc", "wctomb",
                               "mbstowcs", "wcstombs",
                               "string", "cstring", "string.h",                                 # <string.h> & <cstring>
                               "memcpy", "memmove", "memchr", "memcmp", "memset", "strcat",     # <string.h> functions
                               "strncat", "strchr", "strrchr", "strcmp", "strncmp", "strcoll",
                               "strcpy", "strncpy", "strerror", "strlen", "strspn", "strcspn",
                               "strpbrk" ,"strstr", "strtok", "strxfrm",
                               "memccpy", "mempcpy", "strcat_s", "strcpy_s", "strdup",      # <string.h> extension functions
                               "strerror_r", "strlcat", "strlcpy", "strsignal", "strtok_r",
                               "iostream", "istream", "ostream", "fstream", "sstream",      # <iostream> family
                               "iomanip", "iosfwd",
                               "ios", "wios", "streamoff", "streampos", "wstreampos",       # <iostream> types
                               "streamsize", "cout", "cerr", "clog", "cin",
                               "boolalpha", "noboolalpha", "skipws", "noskipws", "showbase",    # <iostream> manipulators
                               "noshowbase", "showpoint", "noshowpoint", "showpos",
                               "noshowpos", "unitbuf", "nounitbuf", "uppercase", "nouppercase",
                               "left", "right", "internal", "dec", "oct", "hex", "fixed",
                               "scientific", "hexfloat", "defaultfloat", "width", "fill",
                               "precision", "endl", "ends", "flush", "ws", "showpoint",
                               "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh",    # <math.h> functions
                               "cosh", "tanh", "exp", "sqrt", "log", "log10", "pow", "powf",
                               "ceil", "floor", "abs", "fabs", "cabs", "frexp", "ldexp",
                               "modf", "fmod", "hypot", "ldexp", "poly", "matherr"]
        self.forbidden_uid = self.__key_words__ + self.__ops__ + self.__macros__ + self.__special_ids__
        e_uids = ["<unk>"]
        for uid in uids:
            if uid in txt2idx.keys() and txt2idx[uid] not in e_uids and uid not in self.forbidden_uid:
                e_uids.append(uid) 
        self.eff_uids = e_uids
        _uids1 = [txt2idx["<unk>"]]
        for uid in uids:
            if uid in txt2idx.keys() and txt2idx[uid] not in _uids1 and uid not in self.forbidden_uid:
                _uids1.append(txt2idx[uid])            
        _uids = []
        # check every subtoken whether or not it can be treated as an valid uid
        for subtoken_idx in range(self.cl.vocab_size):
            subtoken = self.cl.tokenizer.convert_ids_to_tokens(subtoken_idx)
            # print("subtoken_idx=",subtoken)
            assert isinstance(subtoken, str)
            if subtoken in [self.cl.tokenizer.bos_token, self.cl.tokenizer.eos_token,
                                self.cl.tokenizer.sep_token, self.cl.tokenizer.pad_token,
                                self.cl.tokenizer.unk_token, self.cl.tokenizer.cls_token,
                                self.cl.tokenizer.mask_token]:
                continue
            if not subtoken.startswith('Ġ'):
                continue
            clear_subtoken = subtoken[1:]
            if clear_subtoken=="":
                continue
            if clear_subtoken[0] in '0987654321':
                continue

            for uid in uids:
                if uid in self.txt2idx.keys() and \
                   clear_subtoken in uid and \
                   uid not in self.forbidden_uid and \
                   subtoken_idx not in _uids and \
                   clear_subtoken not in self.forbidden_uid:
                    _uids.append(subtoken_idx)
                    # print("_uids=",_uids)
                    break
        
        self._uids = _uids
        self._uids1 = _uids1
        #print([self.cl.tokenizer.convert_ids_to_tokens(i) for i in self._uids])
        #input()
        self.uids = self.__gen_uid_mask_on_vocab(_uids)
        self.uids1 = self.__gen_uid_mask_on_vocab(_uids1)
        
    def __gen_uid_mask_on_vocab(self, uids):
    
        _uids = torch.zeros(self.cl.vocab_size)
        _uids.index_put_([torch.LongTensor(uids)], torch.Tensor([1 for _ in uids]))
        _uids = _uids.reshape([self.cl.vocab_size, 1]).to(self.cl.device)
        return _uids
    
    def __gen_uid_mask_on_seq(self, uids):
        
        _uids = torch.zeros(self.cl.max_len)
        _uids.index_put_([torch.LongTensor(uids)], torch.Tensor([1 for _ in uids]))
        _uids = _uids.reshape([self.cl.max_len, 1]).cuda()
        return _uids
    def __gen_uid_vocab(self, dictionary):  
        _uids1 = torch.zeros(self.cl.vocab_size)   
        for idx, val in dictionary.items():  
            _uids1[idx] = val  
        #print("_uids[158]=",_uids1[158])
        _uids1 = _uids1.view(-1, 1)   
        if torch.cuda.is_available():  
             _uids1 = _uids1.cuda()  
        return _uids1
 
    def rename_uid(self,uids, x, y, x_uid,vocab_cnt_test, ori_uid, n_candidate=5):
        # print("x_raw=,y=",x, y)
        _x_uid = []
        for i in x_uid:
            if i < self.cl.max_len:
                _x_uid.append(i)
        x_uid = self.__gen_uid_mask_on_seq(_x_uid)
        ori_uid_id =ori_uid 
        
        if ori_uid in self.txt2idx.keys():
            ori_uid = self.txt2idx[ori_uid]
        else:
            ori_uid = self.txt2idx['<unk>']
        
        eff = self.eff_uids 
        vocab_dict = {token: value for token, value in vocab_cnt_test}  
        eff_values = [vocab_dict.get(token, 0) for token in eff]  

        value_ori = 0  
        for item in vocab_cnt_test:
             if item[0] == ori_uid_id:
                 value_ori = item[1]
                 break     
        normalized_values_ori= [(1-abs(value-value_ori)/(value+value_ori)) if (value-value_ori) != 0 else 0 for value in eff_values]

        normalized_values_ori1 =[-x * math.log2(x) if x > 0 else 0 for x in normalized_values_ori]
      
        UID=self._uids1
        #dic_uid =torch.zeros(4831) 
        dic_uid = dict([(k, v) for k, v in zip(UID, normalized_values_ori1)])
        tt=self.__gen_uid_vocab(dic_uid)#torch.Size([5000, 1])

        delta_embed = self.uids1 *(tt)
        inner_prod = torch.sum(delta_embed, dim=1) 
        _, new_uid_cand =  torch.topk(inner_prod, n_candidate)
        new_uid_cand = new_uid_cand.cpu().numpy()
        new_x, new_x_uid = [], []
        for Gnew_uid1 in new_uid_cand:                  
            for token, token_id in self.txt2idx.items():
                if token_id == Gnew_uid1:
                    Gnew_uid=token
                    break
            if Gnew_uid in x:
                continue
                print("Gnew_uid in x")
            new_x_uid.append(Gnew_uid)
            new_x.append(copy.deepcopy(x))
            for i in range(len(new_x[-1])):
                if new_x[-1][i] == ori_uid_id:
                    new_x[-1][i] = Gnew_uid 
            try:
                parser.parse(" ".join(new_x[-1]))
            except:
                new_x_uid.pop()
                new_x.pop()
        if len(new_x_uid) == 0:
            return None, None
        while len(new_x_uid) < n_candidate:
            new_x_uid.append(new_x_uid[-1])
            new_x.append(new_x[-1])
        count1 = 0  
        for sublist in new_x:
            if sublist == x:  
                count1 += 1
        return new_x, new_x_uid
  
class InsModifier(object):   
    def __init__(self, classifier, txt2idx, idx2txt, poses=None):
        
        self.cl = classifier
        self.txt2idx = txt2idx
        self.idx2txt = idx2txt
        if poses is not None: # else you need to call initInsertDict later
          self.initInsertDict(poses)
    
    def _insert2idxs(self, insert):
        idxs = []
        for t in insert:
            if self.txt2idx.get(t) is not None:
              idxs.append(self.txt2idx[t])
            else:
              idxs.append(self.txt2idx['<unk>'])
        return idxs
    def initInsertDict(self, poses):
        self.insertDict = dict([(pos, []) for pos in poses])
        self.insertDict["count"] = 0

    def interpolate_attack(self, X, tok):
        if not X:
            return []
        X = list(X)
        if tok is None:
            tok_seq = []
        elif len(tok) > 0 and isinstance(tok[0], list):
            tok_seq = tok[0]
        else:
            tok_seq = tok
        tok_seq = list(tok_seq)
        functions = []
        CONTROL_KEYWORDS = {
            "if", "for", "while", "switch", "catch",
            "return", "sizeof", "do", "else", "case"
        }
        DECL_BOUNDARIES = {";", "{", "}", ":"}
        def is_identifier(t):
            return (
                isinstance(t, str)
                and t != ""
                and (t[0].isalpha() or t[0] == "_")
                and all(c.isalnum() or c == "_" for c in t)
            )

        def find_match(start, left, right):
            if start < 0 or start >= len(X) or X[start] != left:
                return -1
            depth = 0
            for k in range(start, len(X)):
                if X[k] == left:
                    depth += 1
                elif X[k] == right:
                    depth -= 1
                    if depth == 0:
                        return k
            return -1

        def contains_subseq(seq, sub):
            if not seq or not sub or len(seq) < len(sub):
                return False
            n = len(seq)
            m = len(sub)
            for s in range(n - m + 1):
                if seq[s:s + m] == sub:
                    return True
            return False

        def find_decl_start(name_idx):
            j = name_idx - 1
            if j < 0:
                return -1
            while j >= 0:
                if X[j] in DECL_BOUNDARIES:
                    return j + 1
                j -= 1
            return 0

        def valid_function_header(start, name_idx):
            if start < 0 or start >= name_idx:
                return False
            prefix = X[start:name_idx]
            if not prefix:
                return False
            if any(t in CONTROL_KEYWORDS for t in prefix):
                return False
            for t in prefix:
                if (
                    is_identifier(t)
                    or t in {
                        "*", "&", "::", "<", ">",
                        "const", "static", "inline",
                        "unsigned", "signed", "long", "short",
                        "struct", "class", "enum"
                    }
                ):
                    return True
            return False
        i = 0
        while i < len(X) - 2:
            if not is_identifier(X[i]):
                i += 1
                continue
            func_name = X[i]
            if func_name in CONTROL_KEYWORDS:
                i += 1
                continue
            if func_name == "main":
                i += 1
                continue
            if i + 1 >= len(X) or X[i + 1] != "(":
                i += 1
                continue
            name_idx = i
            param_end = find_match(name_idx + 1, "(", ")")
            if param_end == -1:
                i += 1
                continue
            body_start = param_end + 1
            while body_start < len(X) and X[body_start] in {
                "const", "noexcept", "override", "final"
            }:
                body_start += 1
                if body_start < len(X) and X[body_start] == "(":
                    tmp_end = find_match(body_start, "(", ")")
                    if tmp_end == -1:
                        break
                    body_start = tmp_end + 1
            if body_start >= len(X) or X[body_start] != "{":
                i += 1
                continue
            body_end = find_match(body_start, "{", "}")
            if body_end == -1:
                i += 1
                continue
            func_start = find_decl_start(name_idx)
            if not valid_function_header(func_start, name_idx):
                i += 1
                continue
            function_tokens = X[func_start:body_end + 1]
            if not contains_subseq(tok_seq, function_tokens):
                functions.append((func_name, function_tokens))
            i = body_end + 1
        if not functions:
            return []
        min_func = min(functions, key=lambda x: len(x[1]))
        return min_func[1]

    #     return []
    def extract_if_structure(self, X):
        if not X:
            return []
        structures = []

        def find_match(tokens, start, left, right):
            if start < 0 or start >= len(tokens) or tokens[start] != left:
                return -1
            depth = 0
            for k in range(start, len(tokens)):
                if tokens[k] == left:
                    depth += 1
                elif tokens[k] == right:
                    depth -= 1
                    if depth == 0:
                        return k
            return -1

        def parse_statement(start):
            if start < 0 or start >= len(X):
                return -1
            tok = X[start]
            if tok == "{":
                end = find_match(X, start, "{", "}")
                return end + 1 if end != -1 else -1
            if tok == ";":
                return start + 1
            if tok == "if":
                return parse_if(start, include_else=True)

            # for / while / switch
            if tok in ("for", "while", "switch"):
                if start + 1 >= len(X) or X[start + 1] != "(":
                    return -1
                cond_end = find_match(X, start + 1, "(", ")")
                if cond_end == -1:
                    return -1

                # switch 
                if tok == "switch":
                    body_start = cond_end + 1
                    if body_start < len(X) and X[body_start] == "{":
                        return parse_statement(body_start)
                    return -1

                # for / while 
                return parse_statement(cond_end + 1)

            # do ... while (...);
            if tok == "do":
                body_end = parse_statement(start + 1)
                if body_end == -1:
                    return -1

                if (
                    body_end < len(X)
                    and X[body_end] == "while"
                    and body_end + 1 < len(X)
                    and X[body_end + 1] == "("
                ):
                    cond_end = find_match(X, body_end + 1, "(", ")")
                    if (
                        cond_end != -1
                        and cond_end + 1 < len(X)
                        and X[cond_end + 1] == ";"
                    ):
                        return cond_end + 2

                return -1
            paren = 0
            bracket = 0
            brace = 0
            for k in range(start, len(X)):
                t = X[k]
                if t == "(":
                    paren += 1
                elif t == ")":
                    if paren == 0:
                        return -1
                    paren -= 1
                elif t == "[":
                    bracket += 1
                elif t == "]":
                    if bracket == 0:
                        return -1
                    bracket -= 1
                elif t == "{":
                    brace += 1
                elif t == "}":
                    if brace == 0:
                        return -1
                    brace -= 1
                elif t == ";" and paren == 0 and bracket == 0 and brace == 0:
                    return k + 1
            return -1

        def parse_if(start, include_else):
            if start < 0 or start >= len(X) or X[start] != "if":
                return -1
            if start + 1 >= len(X) or X[start + 1] != "(":
                return -1
            cond_end = find_match(X, start + 1, "(", ")")
            if cond_end == -1:
                return -1
            body_start = cond_end + 1
            body_end = parse_statement(body_start)
            if body_end == -1:
                return -1
            if include_else and body_end < len(X) and X[body_end] == "else":
                if body_end + 1 >= len(X):
                    return -1
                if X[body_end + 1] == "if":
                    else_end = parse_if(body_end + 1, include_else=True)
                else:
                    else_end = parse_statement(body_end + 1)
                if else_end == -1:
                    return -1
                body_end = else_end
            return body_end

        def replace_condition_false(stmt):
            if len(stmt) < 4 or stmt[0] != "if" or stmt[1] != "(":
                return []
            cond_end = find_match(stmt, 1, "(", ")")
            if cond_end == -1:
                return []
            return stmt[:2] + ["1", "==", "0"] + stmt[cond_end:]
        for i, tok in enumerate(X):
            if tok != "if":
                continue
            end = parse_if(i, include_else=False)
            if end == -1:
                continue
            stmt = X[i:end]
            if stmt:
                structures.append(stmt)
        if not structures:
            return []
        min_structure = min(structures, key=len)
        return min_structure

    def extract_loop_structure(self, X):
        if not X:
            return []
        loops = []
        do_tail_while_indices = set()
        def find_match(tokens, start, left, right):
            if start < 0 or start >= len(tokens) or tokens[start] != left:
                return -1
            depth = 0
            for k in range(start, len(tokens)):
                if tokens[k] == left:
                    depth += 1
                elif tokens[k] == right:
                    depth -= 1
                    if depth == 0:
                        return k
            return -1

        def parse_if(start, include_else=True):
            if start < 0 or start >= len(X) or X[start] != "if":
                return -1
            if start + 1 >= len(X) or X[start + 1] != "(":
                return -1
            cond_end = find_match(X, start + 1, "(", ")")
            if cond_end == -1:
                return -1
            body_end = parse_statement(cond_end + 1)
            if body_end == -1:
                return -1
            if include_else and body_end < len(X) and X[body_end] == "else":
                if body_end + 1 >= len(X):
                    return -1
                if X[body_end + 1] == "if":
                    else_end = parse_if(body_end + 1, include_else=True)
                else:
                    else_end = parse_statement(body_end + 1)
                if else_end == -1:
                    return -1
                body_end = else_end
            return body_end

        def parse_statement(start):
            if start < 0 or start >= len(X):
                return -1
            tok = X[start]
            #  { ... }
            if tok == "{":
                end = find_match(X, start, "{", "}")
                return end + 1 if end != -1 else -1
            # 
            if tok == ";":
                return start + 1
            # 3：if / if-else 
            if tok == "if":
                return parse_if(start, include_else=True)
            # for / while / switch
            if tok in ("for", "while", "switch"):
                if start + 1 >= len(X) or X[start + 1] != "(":
                    return -1
                cond_end = find_match(X, start + 1, "(", ")")
                if cond_end == -1:
                    return -1
                if tok == "switch":
                    body_start = cond_end + 1
                    if body_start < len(X) and X[body_start] == "{":
                        return parse_statement(body_start)
                    return -1
                return parse_statement(cond_end + 1)

            # do ... while (...);
            if tok == "do":
                body_end = parse_statement(start + 1)
                if body_end == -1:
                    return -1

                if (
                    body_end < len(X)
                    and X[body_end] == "while"
                    and body_end + 1 < len(X)
                    and X[body_end + 1] == "("
                ):
                    cond_end = find_match(X, body_end + 1, "(", ")")
                    if (
                        cond_end != -1
                        and cond_end + 1 < len(X)
                        and X[cond_end + 1] == ";"
                    ):
                        return cond_end + 2

                return -1

            paren = 0
            bracket = 0
            brace = 0
            for k in range(start, len(X)):
                t = X[k]
                if t == "(":
                    paren += 1
                elif t == ")":
                    if paren == 0:
                        return -1
                    paren -= 1
                elif t == "[":
                    bracket += 1
                elif t == "]":
                    if bracket == 0:
                        return -1
                    bracket -= 1
                elif t == "{":
                    brace += 1
                elif t == "}":
                    if brace == 0:
                        return -1
                    brace -= 1
                elif t == ";" and paren == 0 and bracket == 0 and brace == 0:
                    return k + 1
            return -1

        def parse_while(start):
            if start + 1 >= len(X) or X[start + 1] != "(":
                return None
            cond_end = find_match(X, start + 1, "(", ")")
            if cond_end == -1:
                return None
            body_start = cond_end + 1
            body_end = parse_statement(body_start)
            if body_end == -1:
                return None
            return {
                "kind": "while",
                "start": start,
                "end": body_end,
                "body_start": body_start,
                "body_end": body_end,
            }

        def parse_for(start):
            if start + 1 >= len(X) or X[start + 1] != "(":
                return None
            header_end = find_match(X, start + 1, "(", ")")
            if header_end == -1:
                return None
            semi_count = 0
            paren = 0
            bracket = 0
            brace = 0
            for k in range(start + 2, header_end):
                t = X[k]
                if t == "(":
                    paren += 1
                elif t == ")":
                    if paren > 0:
                        paren -= 1
                elif t == "[":
                    bracket += 1
                elif t == "]":
                    if bracket > 0:
                        bracket -= 1
                elif t == "{":
                    brace += 1
                elif t == "}":
                    if brace > 0:
                        brace -= 1
                elif t == ";" and paren == 0 and bracket == 0 and brace == 0:
                    semi_count += 1
            if semi_count < 2:
                return None
            body_start = header_end + 1
            body_end = parse_statement(body_start)
            if body_end == -1:
                return None
            return {
                "kind": "for",
                "start": start,
                "end": body_end,
                "body_start": body_start,
                "body_end": body_end,
            }

        def parse_do(start):
            body_start = start + 1
            body_end = parse_statement(body_start)
            if body_end == -1:
                return None
            if not (
                body_end < len(X)
                and X[body_end] == "while"
                and body_end + 1 < len(X)
                and X[body_end + 1] == "("
            ):
                return None
            cond_end = find_match(X, body_end + 1, "(", ")")
            if cond_end == -1:
                return None
            if cond_end + 1 >= len(X) or X[cond_end + 1] != ";":
                return None
            return {
                "kind": "do",
                "start": start,
                "end": cond_end + 2,
                "body_start": body_start,
                "body_end": body_end,
                "tail_while": body_end,
            }
        
        def make_false_loop(info):
            return X[info["start"]:info["end"]]
        for i, tok in enumerate(X):
            info = None
            if tok == "do":
                info = parse_do(i)
                if info is not None:
                    do_tail_while_indices.add(info["tail_while"])
            elif tok == "while":
                if i in do_tail_while_indices:
                    continue

                info = parse_while(i)

            elif tok == "for":
                info = parse_for(i)

            if info is not None:
                loop_code = make_false_loop(info)
                if loop_code:
                    loops.append(loop_code)
        if not loops:
            return []
        return min(loops, key=len)

    def extract_variable_statements(self, X):
        if not X:
            return []

        IO_FUNCS = {"scanf", "printf", "gets", "puts"}

        def is_identifier(tok):
            return (
                isinstance(tok, str)
                and tok != ""
                and (tok[0].isalpha() or tok[0] == "_")
                and all(c.isalnum() or c == "_" for c in tok)
            )

        def find_match(start, left, right):
            if start < 0 or start >= len(X) or X[start] != left:
                return -1
            depth = 0
            for k in range(start, len(X)):
                if X[k] == left:
                    depth += 1
                elif X[k] == right:
                    depth -= 1
                    if depth == 0:
                        return k
            return -1

        def find_stmt_end(start):
            paren = 0
            bracket = 0
            brace = 0
            for k in range(start, len(X)):
                t = X[k]
                if t == "(":
                    paren += 1
                elif t == ")":
                    if paren == 0:
                        return -1
                    paren -= 1
                elif t == "[":
                    bracket += 1
                elif t == "]":
                    if bracket == 0:
                        return -1
                    bracket -= 1
                elif t == "{":
                    brace += 1
                elif t == "}":
                    if brace == 0:
                        return -1
                    brace -= 1
                elif t == ";" and paren == 0 and bracket == 0 and brace == 0:
                    return k + 1
            return -1

        def is_var_decl(stmt):
            if not stmt or stmt[-1] != ";":
                return False
            if stmt[0] in {"typedef", "using"}:
                return False
            i = 0
            saw_type = False
            while i < len(stmt) - 1:
                t = stmt[i]
                if t in {
                    "const", "static", "extern", "register", "volatile",
                    "auto", "restrict", "mutable"
                }:
                    i += 1
                    continue
                if t in {
                    "signed", "unsigned",
                    "int", "float", "double", "char",
                    "long", "short", "bool", "string", "size_t"
                }:
                    saw_type = True
                    i += 1
                    continue
                # struct / enum / class 
                if t in {"struct", "enum", "class"}:
                    if i + 1 < len(stmt) - 1 and is_identifier(stmt[i + 1]):
                        saw_type = True
                        i += 2
                        continue
                    return False
                break
            if not saw_type:
                return False
            j = i
            while j < len(stmt) - 1 and stmt[j] in {"*", "&"}:
                j += 1
            if j >= len(stmt) - 1 or not is_identifier(stmt[j]):
                return False
            if j + 1 < len(stmt) - 1 and stmt[j + 1] == "(":
                return False

            return True

        def is_scanf_printf(stmt):
            return (
                len(stmt) >= 4
                and stmt[0] in IO_FUNCS
                and stmt[1] == "("
                and stmt[-1] == ";"
            )

        def is_cin_cout(stmt):
            return (
                len(stmt) >= 3
                and stmt[0] in {"cin", "cout"}
                and stmt[-1] == ";"
                and any(t in {">>", "<<"} for t in stmt)
            )
        result = []
        i = 0
        while i < len(X):
            t = X[i]
            if t in {"{", "}"}:
                i += 1
                continue
            if t in {"case", "default"}:
                j = i + 1
                while j < len(X) and X[j] != ":":
                    j += 1
                i = j + 1 if j < len(X) else i + 1
                continue
            if (
                t in {"if", "for", "while", "switch"}
                and i + 1 < len(X)
                and X[i + 1] == "("
            ):
                end = find_match(i + 1, "(", ")")
                i = end + 1 if end != -1 else i + 1
                continue
            if t == "do":
                i += 1
                continue
            end = find_stmt_end(i)
            if end == -1:
                i += 1
                continue
            stmt = X[i:end]
            if is_var_decl(stmt) or is_scanf_printf(stmt) or is_cin_cout(stmt):
                result.extend(stmt)
            i = end
        return result

    def wrap_block(self, inst, context_tokens=None, guard_name="check_flag_0"):
        """
        Turn the C token into dead code
        """
        if not inst:
            return []

        def is_identifier(t):
            return (
                isinstance(t, str)
                and t != ""
                and (t[0].isalpha() or t[0] == "_")
                and all(c.isalnum() or c == "_" for c in t)
            )
        def is_unsafe_c_fragment(tokens):
            if not tokens:
                return True
            if "case" in tokens or "default" in tokens:
                return True
            if tokens[0] in {"break", "continue"}:
                return True
            
            for i in range(len(tokens) - 1):
                if (
                    is_identifier(tokens[i])
                    and tokens[i + 1] == ":"
                    and tokens[i] not in {"case", "default"}
                ):
                    return True

            if len(tokens) >= 4:
                first = tokens[0]
                if (
                    first not in {
                        "if", "for", "while", "switch", "do",
                        "else", "typedef", "struct", "union", "enum"
                    }
                    and "(" in tokens
                    and ")" in tokens
                    and "{" in tokens
                ):
                    # int foo ( int x ) { ... }
                    # void solve ( ) { ... }
                    # static int foo ( ) { ... }
                    try:
                        lpar = tokens.index("(")
                        rpar = tokens.index(")")
                        lbrace = tokens.index("{")
                        if lpar < rpar < lbrace:
                            return True
                    except ValueError:
                        pass
            return False

        def pick_natural_guard_name(tokens):
            preferred_names = [
                "debug_flag", "check_flag", "valid_flag", "ready_flag",
                "enable_check", "need_check", "skip_flag", "active_flag",
                "allow_flag", "guard_flag", "check_flag_0"
            ]
            candidates = []
            if tokens:
                for t in tokens:
                    if is_identifier(t) and t in preferred_names:
                        candidates.append(t)
                if candidates:
                    candidates = list(set(candidates))
                    return random.choice(candidates)
                
                for i in range(len(tokens) - 2):
                    t = tokens[i]
                    nxt = tokens[i + 1]
                    rhs = tokens[i + 2]
                    if not is_identifier(t):
                        continue
                    if len(t) < 4:
                        continue
                    lower_t = t.lower()
                    looks_like_flag = (
                        "flag" in lower_t
                        or "check" in lower_t
                        or "valid" in lower_t
                        or "ready" in lower_t
                        or "enable" in lower_t
                        or "active" in lower_t
                        or "allow" in lower_t
                        or "skip" in lower_t
                    )
                    if not looks_like_flag:
                        continue
                    if nxt == "=" and rhs in {
                        "0", "1", "false", "true",
                        "FALSE", "TRUE", "False", "True"
                    }:
                        candidates.append(t)
                if candidates:
                    candidates = list(set(candidates))
                    return random.choice(candidates)
            return random.choice(preferred_names)

        if is_unsafe_c_fragment(inst):
            return []
        guard_name = pick_natural_guard_name(context_tokens)
        false_guard_templates = [
            # if (guard_name)
            ["if", "(", guard_name, ")", "{"],
            # if (guard_name != 0)
            ["if", "(", guard_name, "!=", "0", ")", "{"],
            # if (guard_name > 0)
            ["if", "(", guard_name, ">", "0", ")", "{"],
            # if (0 && guard_name)
            ["if", "(", "0", "&&", guard_name, ")", "{"],
            # if (guard_name && 0)
            # guard_name is 0
            ["if", "(", guard_name, "&&", "0", ")", "{"],
            # if (!1 && guard_name)
            ["if", "(", "!", "1", "&&", guard_name, ")", "{"],
            # if ((guard_name) && (1 < 0))
            ["if", "(", "(", guard_name, ")", "&&", "(", "1", "<", "0", ")", ")", "{"],
            # if ((guard_name + 0) != 0)
            ["if", "(", "(", guard_name, "+", "0", ")", "!=", "0", ")", "{"],
        ]
        prefix = random.choice(false_guard_templates)
        return (
            ["{",
            "int", guard_name, "=", "0", ";"] +
            prefix +
            inst +
            ["}", "}"]
        )

    def trim_c_inst_by_complete_stmt(self, inst, max_tokens=50):
        if not inst:
            return []
        if len(inst) <= max_tokens:
            return inst
        open_to_close = {"(": ")", "[": "]", "{": "}"}
        close_to_open = {")": "(", "]": "[", "}": "{"}
        stack = []
        candidates = []
        i = 0
        while i < len(inst):
            t = inst[i]
            if t in {'"', "'"}:
                q = t
                i += 1
                while i < len(inst) and inst[i] != q:
                    i += 1
                i += 1
                continue
            if t in open_to_close:
                stack.append(t)
            elif t in close_to_open:
                if stack and stack[-1] == close_to_open[t]:
                    stack.pop()
            has_open_paren_or_bracket = any(s in {"(", "["} for s in stack)
            if not has_open_paren_or_bracket:
                if t == ";":
                    candidates.append((i + 1, list(stack)))
                elif t == "}":
                    if i + 1 < len(inst) and inst[i + 1] == "else":
                        pass
                    else:
                        candidates.append((i + 1, list(stack)))
            i += 1
        for cut, st in reversed(candidates):
            need_close_braces = sum(1 for s in st if s == "{")
            if cut + need_close_braces <= max_tokens:
                return inst[:cut] + ["}"] * need_close_braces
        return []
    
    def insert4(self,x_token, x,rand_sa, n_candidate=5):
        pos_candidates = patternpy.InsAddCandidates(self.insertDict, self.cl.max_len) # exclude outlier poses
        n = len(pos_candidates)
        tok=x_token[0] 
        # numb=50
        numb=70           
        if n_candidate < n:
          candisIdx = random.sample(range(n), n_candidate)
        else:
          candisIdx = random.sample(range(n), n)
        pos_candidates = [pos_candidates[candiIds] for candiIds in candisIdx] # sample max(n, n_candidate) poses
        new_x, new_insertDict = [], [] 
        for pos in pos_candidates:          
            rand1= random.choice(rand_sa)      
            rand_d1=rand1['raw'][0]
            fragment_types = ['function', 'comment', 'selection', 'loop']
            max_attempts = 10
            attempt_count = 0  
            while attempt_count < max_attempts:
                attempt_count = attempt_count + 1
                selected_type = random.choice(fragment_types)
                if selected_type == 'function':
                    inst = self.interpolate_attack(rand_d1, tok)  
                    if inst:  
                       if len(inst) > numb:
                            inst = self.trim_c_inst_by_complete_stmt(inst, max_tokens=numb)
                            if not inst:
                                continue                         
                       inst_idxs = self._insert2idxs(inst)
                       _insertDict = deepcopy(self.insertDict)
                       suc=patternpy.InsAdd(_insertDict, pos, inst_idxs)
                       if not suc:
                          continue
                       _x = patternpy.InsResult1(x,pos, _insertDict)
                       new_x.append(_x)
                       new_insertDict.append(_insertDict)
                       return new_x, new_insertDict
                    else:  
                        if 'function' in fragment_types:
                            fragment_types.remove('function')

                        if not fragment_types:
                            break
                elif selected_type == 'comment':
                    inst = self.extract_variable_statements(rand_d1)  
                    if inst:
                        break
                elif selected_type == 'selection':
                    inst = self.extract_if_structure(rand_d1)  
                    if inst:
                        break
                elif selected_type == 'loop':
                    inst = self.extract_loop_structure(rand_d1)  
                    if inst:
                        break   
            _insertDict = deepcopy(self.insertDict)
            guard_name = "i"
            inst = self.wrap_block(inst, guard_name)
            if len(inst) > numb:
                inst = self.trim_c_inst_by_complete_stmt(inst, max_tokens=numb)
                if not inst:
                    continue            
            inst_idxs = self._insert2idxs(inst)                          
            suc=patternpy.InsAdd(_insertDict, pos, inst_idxs)
            if not suc:
                continue
            _x = patternpy.InsResult(x, _insertDict)
            new_x.append(_x)            
            new_insertDict.append(_insertDict)
        return new_x, new_insertDict    
 
    def remove(self, x_raw, n_candidate=5):

        pos_candidates = patternpy.InsDeleteCandidates(self.insertDict) # e.g. [(pos0, 0), (pos0, 1), (pos1, 0), ...]
        n = len(pos_candidates)
        if n_candidate < n:
          candisIdx = random.sample(range(n), n_candidate)
        else:
          candisIdx = random.sample(range(n), n)
        pos_candidates = [pos_candidates[candiIds] for candiIds in candisIdx]

        new_x_raw, new_insertDict = [], []
        for pos, listIdx in pos_candidates:
            _insertDict = deepcopy(self.insertDict)
            patternpy.InsDelete(_insertDict, pos, listIdx)
            _x_raw = patternpy.InsResult(x_raw, _insertDict)
            try:
                parser.parse(" ".join(_x_raw))
            except:
                continue
            new_x_raw.append(_x_raw)
            new_insertDict.append(_insertDict)

        return new_x_raw, new_insertDict

    def insert_remove_random(self, x_raw):

        new_x_raw, new_insertDict = [], []
        fail_cnt = 0
        while True:
            if fail_cnt >= 10:  # in case of dead loop
                break
            if random.random() > 0.5: # insert
                pos_candidates = patternpy.InsAddCandidates(self.insertDict)
                if pos_candidates == []:
                    fail_cnt += 1
                    continue
                pos_cand = random.sample(pos_candidates, 1)[0]
                inst = random.sample(self.inserts, 1)[0]
                _insertDict = deepcopy(self.insertDict)
                patternpy.InsAdd(_insertDict, pos_cand, inst)
            else:
                pos_candidates = patternpy.InsDeleteCandidates(self.insertDict)
                if pos_candidates == []:
                    fail_cnt += 1
                    continue
                pos_cand, inPosIdx = random.sample(pos_candidates, 1)[0]
                _insertDict = deepcopy(self.insertDict)
                patternpy.InsDelete(_insertDict, pos_cand, inPosIdx)
            _x_raw = patternpy.InsResult(x_raw, _insertDict)
            try:
                parser.parse(" ".join(_x_raw))
            except:
                fail_cnt += 1
                continue
            new_x_raw.append(_x_raw)
            new_insertDict.append(_insertDict)
            break
        return new_x_raw, new_insertDict

