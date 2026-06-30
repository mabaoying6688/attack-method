# -*- coding: utf-8 -*-
import zlib
import re
import torch
import copy
import sys
import math
import numpy as np
from collections import deque 
from preprocesslstm import mytoken
from preprocesslstm import patternpy
import random
from copy import deepcopy
from torch.nn.functional import cosine_similarity

class TokenModifier(object):
    
    def __init__(self, classifier, loss, uids, txt2idx, idx2txt):
        
        self.cl = classifier
        self.loss = loss
        self.txt2idx = txt2idx
        self.idx2txt = idx2txt
        self.__key_words__ = [ 'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 
                            'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 
                            'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 
                            'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 
                            'try', 'while', 'with', 'yield']

        self.__builtins__ = ['abs', 'all', 'any', 'ascii', 'bin', 'bool', 'breakpoint', 'bytearray', 
                        'bytes', 'callable', 'chr', 'classmethod', 'compile', 'complex', 
                        'delattr', 'dict', 'dir', 'divmod', 'enumerate', 'eval', 'exec', 
                        'filter', 'float', 'format', 'frozenset', 'getattr', 'globals', 
                        'hasattr', 'hash', 'help', 'hex', 'id', 'input', 'int', 'isinstance', 
                        'issubclass', 'iter', 'len', 'list', 'locals', 'map', 'max', 'memoryview', 
                        'min', 'next', 'object', 'oct', 'open', 'ord', 'pow', 'print', 'property', 
                        'range', 'repr', 'reversed', 'round', 'set', 'setattr', 'slice', 
                        'sorted', 'staticmethod', 'str', 'sum', 'super', 'tuple', 'type', 
                        'vars', 'zip', '__import__']

        self.__magic_methods__=[
                        '__abs__', '__add__', '__and__', '__bool__', '__ceil__', '__class__', 
                        '__delattr__', '__dict__', '__dir__', '__divmod__', '__doc__', 
                        '__eq__', '__float__', '__floor__', '__floordiv__', '__format__', 
                        '__ge__', '__getattribute__', '__getitem__', '__getnewargs__', 
                        '__gt__', '__hash__', '__index__', '__init__', '__init_subclass__', 
                        '__int__', '__invert__', '__iter__', '__le__', '__len__', '__lshift__', 
                        '__lt__', '__mod__', '__mul__', '__ne__', '__neg__', '__new__', 
                        '__or__', '__pos__', '__pow__', '__radd__', '__rand__', '__rdivmod__', 
                        '__reduce__', '__reduce_ex__', '__repr__', '__rfloordiv__', '__rlshift__', 
                        '__rmod__', '__rmul__', '__ror__', '__round__', '__rpow__', '__rrshift__', 
                        '__rshift__', '__rsub__', '__rtruediv__', '__rxor__', '__setattr__', 
                        '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__truediv__', 
                        '__xor__', '__import__', '__name__', '__qualname__', '__module__', 
                        '__annotations__', '__bases__', '__mro__', '__call__', '__closure__', 
                        '__code__', '__defaults__', '__kwdefaults__', '__globals__']

        self.__stdlib__ = [
                            
                            'sys', 'os', 're', 'json', 'math', 'random', 'datetime', 'time', 
                            'collections', 'itertools', 'functools', 'threading', 'multiprocessing', 
                            'subprocess', 'argparse', 'logging', 'unittest', 'pytest', 'numpy', 
                            'pandas', 'matplotlib', 'tensorflow', 'torch', 'flask', 'django',
                            
                            
                            'open', 'exit', 'quit', 'help', 'dir', 'type', 'isinstance', 
                            'hasattr', 'getattr', 'setattr', 'property', 'staticmethod', 
                            'classmethod', 'super', 'enumerate', 'zip', 'map', 'filter', 
                            'reduce', 'sorted', 'reversed', 'range', 'slice', 'memoryview']
      
        self.__ops__ = ['+', '-', '*', '/', '%', '**', '//', '<<', '>>', '&', '|', '^', 
                        '~', '<', '>', '<=', '>=', '==', '!=', 'and', 'or', 'not', 'is', 
                        'in', ':=']
      
        self.__special_vars__ = ['Ellipsis', 'NotImplemented', '__debug__',  '__main__', 
                        '__name__', '__file__', '__package__', '__doc__', '__annotations__', 
                        '__builtins__', '__loader__', '__spec__', '__path__', '__cached__']

       
        self.forbidden_uid = self.__key_words__ + self.__builtins__+ self.__magic_methods__+ self.__stdlib__+self.__ops__+ self.__special_vars__
        e_uids = ["<unk>"]
        for uid in uids:
            if uid in txt2idx.keys() and txt2idx[uid] not in e_uids and uid not in self.forbidden_uid:
                e_uids.append(uid) 
        self.eff_uids = e_uids

        _uids = [txt2idx["<unk>"]]
        
        for uid in uids:
            if uid in txt2idx.keys() and txt2idx[uid] not in _uids and uid not in self.forbidden_uid:
                _uids.append(txt2idx[uid])

        with open("MY-sort-hefaUID.txt", "w", encoding="utf-8") as file:
          for item in _uids:
            file.write(str(item) + "\n")
 
        self._uids = _uids               

        self.uids = self.__gen_uid_mask_on_vocab(_uids)
        

    def __gen_uid_mask_on_vocab(self, uids):
        
        _uids = torch.zeros(self.cl.vocab_size)
        _uids.index_put_([torch.LongTensor(uids)], torch.Tensor([1 for _ in uids]))
        _uids = _uids.reshape([self.cl.vocab_size, 1]).cuda()
        return _uids

    def __gen_uid_mask_on_seq(self, uids):        
        _uids = torch.zeros(self.cl.max_len)
        _uids.index_put_([torch.LongTensor(uids)], torch.Tensor([1 for _ in uids]))
        _uids = _uids.reshape([self.cl.max_len, 1]).cuda()
        return _uids#返回生成的掩码张量
    


    def __gen_uid_vocab1(self, dictionary):
        vocab_size = max(dictionary.keys()) + 1
        _uids1 = torch.zeros(vocab_size, dtype=torch.float32)  
        for idx, val in dictionary.items():
            _uids1[idx] = val
        _uids1 = _uids1.view(-1, 1)
        if torch.cuda.is_available():
            _uids1 = _uids1.cuda()
        return _uids1

    def __gen_uid_vocab(self, dictionary):  
        _uids1 = torch.zeros(self.cl.vocab_size)    
        for idx, val in dictionary.items():  
            _uids1[idx] = val  
        _uids1 = _uids1.view(-1, 1)  

        if torch.cuda.is_available():  
             _uids1 = _uids1.cuda()  
        return _uids1


    def rename_uid(self,uids, x, y, x_uid,vocab_cnt_test, ori_uid, n_candidate=5):
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
        UID=self._uids 
        dic_uid = dict([(k, v) for k, v in zip(UID, normalized_values_ori1)])
        tt=self.__gen_uid_vocab(dic_uid)#torch.Size([5000, 1])
        delta_embed = self.uids *(tt)
        inner_prod = torch.sum(delta_embed, dim=1) 
        _, new_uid_cand =  torch.topk(inner_prod, n_candidate)
        new_uid_cand = new_uid_cand.cpu().numpy()
        new_x = []
        for new_uid in new_uid_cand:
            new_x.append(copy.deepcopy(x[0]))
            for i in _x_uid:
                new_x[-1][i] = new_uid
        return new_x, new_uid_cand
    

class InsModifier(object):
    
    def __init__(self, classifier, txt2idx, poses=None):
        
        self.cl = classifier
        self.txt2idx = txt2idx
        if poses != None: # else you need to call initInsertDict later
          self.initInsertDict(poses)
    def initInsertDict(self, poses):
        self.insertDict = dict([(pos, []) for pos in poses])
        self.insertDict["count"] = 0
    def _insert2idxs(self, insert):
        idxs = []
        for t in insert:
            if self.txt2idx.get(t) is not None:
              idxs.append(self.txt2idx[t])
            else:
              idxs.append(self.txt2idx['<unk>'])
        return idxs
 

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
        LAYOUT = {
            "", " ", "\t", "\r", "\n",
            "NEWLINE", "NL", "<NL>",
            "INDENT", "DEDENT", "<INDENT>", "<DEDENT>"
        }
        QUOTES = {'"', "'"}
        OPEN = {'(': ')', '[': ']', '{': '}'}
        CLOSE = {')': '(', ']': '[', '}': '{'}
        ASSIGN_OPS = {
            '=', '+=', '-=', '*=', '/=', '//=', '%=', '**=',
            '&=', '|=', '^=', '>>=', '<<=', ':='
        }
        COMPOUND_START = {
            'if', 'for', 'while', 'try', 'with', 'def', 'class'
        }
        BRANCH_START = {
            'elif', 'else', 'except', 'finally'
        }
        SIMPLE_KEYWORDS = {
            'return', 'print', 'import', 'from', 'raise', 'assert',
            'break', 'continue', 'pass', 'del', 'global', 'nonlocal', 'yield'
        }
        BLOCK_PREV = {
            '.', '=', '+=', '-=', '*=', '/=', '//=', '%=', '**=',
            '&=', '|=', '^=', '>>=', '<<=', ':=',
            '+', '-', '*', '/', '//', '%', '**',
            '<', '>', '==', '!=', '<=', '>=',
            'and', 'or', 'not', 'in', 'is', 'as',
            'raise', 'return', 'yield', 'assert',
            'from', 'import', 'del', 'lambda',
            ',', ':', '(', '[', '{', '->'
        }
        def normalize_tokens(seq):
            out = []
            if seq is None:
                return out

            for t in seq:
                if t is None:
                    continue
                if not isinstance(t, str):
                    t = str(t)
                if t in LAYOUT:
                    continue
                out.append(t)

            return out
        tokens = normalize_tokens(X)
        tok_norm = normalize_tokens(tok_seq)
        n = len(tokens)
        def contains_subseq(seq, sub):
            if not seq or not sub or len(seq) < len(sub):
                return False
            m = len(sub)
            for i in range(len(seq) - m + 1):
                if seq[i:i + m] == sub:
                    return True
            return False
        def is_identifier(t):
            return (
                isinstance(t, str)
                and t != ""
                and (t[0].isalpha() or t[0] == "_")
                and all(c.isalnum() or c == "_" for c in t)
            )
        def skip_quote(i):
            q = tokens[i]
            j = i + 1

            while j < n:
                if tokens[j] == q:
                    return j + 1
                j += 1
            return i + 1

        def build_quote_mask_and_depths():
            quote_mask = [False] * n
            depths = [0] * n
            depth = 0
            i = 0
            while i < n:
                depths[i] = depth
                t = tokens[i]
                if t in QUOTES:
                    j = skip_quote(i)
                    for k in range(i, min(j, n)):
                        quote_mask[k] = True
                        depths[k] = depth
                    i = j
                    continue
                if t in OPEN:
                    depth += 1
                elif t in CLOSE:
                    depth = max(0, depth - 1)
                i += 1
            return quote_mask, depths
        quote_mask, depths = build_quote_mask_and_depths()

        def probable_statement_boundary_at(i):
            if i <= 0:
                return False
            return tokens[i - 1] not in BLOCK_PREV

        def find_match(start, left, right, stop_on_statement=False):
            if start < 0 or start >= n or tokens[start] != left:
                return -1
            depth = 0
            i = start
            while i < n:
                t = tokens[i]
                if t in QUOTES:
                    i = skip_quote(i)
                    continue
                if t == left:
                    depth += 1
                elif t == right:
                    depth -= 1
                    if depth == 0:
                        return i
                elif stop_on_statement and depth == 1 and i > start:
                    if t in {'def', 'class'} and probable_statement_boundary_at(i):
                        return -1
                i += 1
            return -1

        def scan_top_level_token_before_colon(lo, target):
            depth = 0
            i = lo
            while i < n:
                t = tokens[i]
                if t in QUOTES:
                    i = skip_quote(i)
                    continue
                if t in OPEN:
                    depth += 1
                elif t in CLOSE:
                    depth = max(0, depth - 1)
                elif depth == 0:
                    if t == target:
                        return True
                    if t in {':', ';'}:
                        return False
                i += 1
            return False

        def find_header_colon(start, owner_pos=None):
            depth = 0
            i = start
            ternary_if = False
            owner_tok = tokens[owner_pos] if owner_pos is not None and 0 <= owner_pos < n else None
            while i < n:
                t = tokens[i]
                if t in QUOTES:
                    i = skip_quote(i)
                    continue
                if t in OPEN:
                    depth += 1
                elif t in CLOSE:
                    depth = max(0, depth - 1)
                elif depth == 0:
                    if t == ':':
                        return i
                    if t == ';':
                        return -1
                    if (
                        owner_tok in (COMPOUND_START | BRANCH_START | {'def'})
                        and i > start
                        and t in (COMPOUND_START | BRANCH_START | {'def', 'class'})
                    ):
                        if t == 'if' and scan_top_level_token_before_colon(i + 1, 'else'):
                            ternary_if = True
                        elif t == 'else' and ternary_if:
                            ternary_if = False
                        elif probable_statement_boundary_at(i):
                            return -1
                i += 1
            return -1

        def top_level_token_exists(lo, hi, token):
            depth = 0
            i = lo
            while i < hi:
                t = tokens[i]
                if t in QUOTES:
                    i = skip_quote(i)
                    continue
                if t in OPEN:
                    depth += 1
                elif t in CLOSE:
                    depth = max(0, depth - 1)
                elif t == token and depth == 0:
                    return True
                i += 1
            return False

        def skip_balanced(pos):
            if pos >= n or tokens[pos] not in OPEN:
                return pos + 1
            left = tokens[pos]
            right = OPEN[left]
            depth = 1
            i = pos + 1
            while i < n:
                t = tokens[i]
                if t in QUOTES:
                    i = skip_quote(i)
                    continue
                if t == left:
                    depth += 1
                elif t == right:
                    depth -= 1
                    if depth == 0:
                        return i + 1
                i += 1
            return n

        def assignment_op_after_lhs(pos):
            if pos >= n or quote_mask[pos] or not is_identifier(tokens[pos]):
                return -1
            i = pos + 1
            while i < n:
                if tokens[i] in ASSIGN_OPS:
                    return i
                if tokens[i] == '.' and i + 1 < n and is_identifier(tokens[i + 1]):
                    i += 2
                    continue
                if tokens[i] == '[':
                    i = skip_balanced(i)
                    continue
                if tokens[i] == ',' and i + 1 < n and is_identifier(tokens[i + 1]):
                    i += 2
                    continue
                break
            return -1

        def looks_like_assignment_start(pos):
            return assignment_op_after_lhs(pos) != -1

        def looks_like_call_start(pos):
            if pos >= n or not is_identifier(tokens[pos]):
                return False
            i = pos + 1
            while i + 1 < n and tokens[i] == '.' and is_identifier(tokens[i + 1]):
                i += 2
            return i < n and tokens[i] == '('

        def compound_header_valid(pos, colon):
            if colon == -1 or pos >= n:
                return False
            t = tokens[pos]
            if t == 'for':
                return top_level_token_exists(pos + 1, colon, 'in')
            if t in {'while', 'if', 'with', 'def', 'class', 'elif', 'except'}:
                return pos + 1 < colon
            if t in {'try', 'else', 'finally'}:
                return True
            return False

        def looks_like_compound_start(pos):
            if (
                pos >= n
                or quote_mask[pos]
                or tokens[pos] not in (COMPOUND_START | BRANCH_START)
            ):
                return False
            colon = find_header_colon(pos + 1, pos)
            return compound_header_valid(pos, colon)

        def raw_statement_start(pos):
            if pos >= n or quote_mask[pos]:
                return False
            t = tokens[pos]
            if looks_like_compound_start(pos):
                return True
            if t in SIMPLE_KEYWORDS:
                return True
            if looks_like_assignment_start(pos):
                return True
            if looks_like_call_start(pos):
                return True
            return False

        def split_allowed(pos, start):
            if pos <= start:
                return False
            prev = tokens[pos - 1]
            tok = tokens[pos]
            if prev == '.':
                return False
            if tokens[start] == 'from' and tok == 'import':
                return False
            if prev in BLOCK_PREV:
                return False
            return True

        def parse_simple(pos):
            start = pos
            if pos >= n:
                return pos, 0
            if tokens[pos] in {'break', 'continue', 'pass'}:
                return pos + 1, 1
            depth = 0
            i = pos
            while i < n:
                t = tokens[i]

                if t in QUOTES:
                    i = skip_quote(i)
                    continue
                if t in OPEN:
                    depth += 1
                elif t in CLOSE:
                    depth = max(0, depth - 1)
                elif t == ';' and depth == 0:
                    return i + 1, 1
                if (
                    depth == 0
                    and i > pos
                    and raw_statement_start(i)
                    and split_allowed(i, start)
                ):
                    return i, 1
                i += 1
            return n, 1 if start < n else 0

        def parse_branch(start):
            colon = find_header_colon(start + 1, start)
            if not compound_header_valid(start, colon):
                return start + 1, 1
            body_end, body_lines = parse_stmt(colon + 1)
            return body_end, 1 + body_lines

        def parse_compound(start):
            t = tokens[start]
            colon = find_header_colon(start + 1, start)
            if not compound_header_valid(start, colon):
                return parse_simple(start)
            body_end, body_lines = parse_stmt(colon + 1)
            end = body_end
            lines = 1 + body_lines
            if t == 'if':
                while (
                    end < n
                    and not quote_mask[end]
                    and tokens[end] in {'elif', 'else'}
                    and looks_like_compound_start(end)
                ):
                    b_end, b_lines = parse_branch(end)
                    if b_end <= end:
                        break
                    end = b_end
                    lines += b_lines
                return end, lines
            if t in {'for', 'while'}:
                if (
                    end < n
                    and not quote_mask[end]
                    and tokens[end] == 'else'
                    and looks_like_compound_start(end)
                ):
                    b_end, b_lines = parse_branch(end)
                    if b_end > end:
                        end = b_end
                        lines += b_lines
                return end, lines
            if t == 'try':
                while (
                    end < n
                    and not quote_mask[end]
                    and tokens[end] in {'except', 'else', 'finally'}
                    and looks_like_compound_start(end)
                ):
                    b_end, b_lines = parse_branch(end)
                    if b_end <= end:
                        break
                    end = b_end
                    lines += b_lines
                return end, lines
            return end, lines

        def parse_stmt(pos):
            if pos >= n:
                return pos, 0
            if looks_like_compound_start(pos):
                return parse_compound(pos)
            return parse_simple(pos)

        def parse_def(start):
            if tokens[start] == 'async':
                if (
                    start + 2 >= n
                    or tokens[start + 1] != 'def'
                    or not is_identifier(tokens[start + 2])
                ):
                    return None
                
                actual_start = start
                def_pos = start + 1
                name_pos = start + 2
            elif tokens[start] == 'def':
                if start > 0 and tokens[start - 1] == 'async':
                    return None
                
                if start + 1 >= n or not is_identifier(tokens[start + 1]):
                    return None
                
                actual_start = start
                def_pos = start
                name_pos = start + 1
            else:
                return None
            
            name = tokens[name_pos]
            if name == 'main':
                return None
            
            if name_pos + 1 >= n or tokens[name_pos + 1] != '(':
                return None

            end_params = find_match(
                name_pos + 1,
                '(',
                ')',
                stop_on_statement=True
            )

            if end_params == -1:
                return None

            colon = find_header_colon(end_params + 1, def_pos)
            if colon == -1:
                return None

            body_end, body_lines = parse_stmt(colon + 1)
            if body_end <= colon + 1:
                return None

            func_tokens = tokens[actual_start:body_end]
            if not func_tokens:
                return None
            if contains_subseq(tok_norm, func_tokens):
                return None
            logical_lines = 1 + body_lines
            return (
                name,
                func_tokens,
                logical_lines,
                len(func_tokens),
                actual_start
            )
        functions = []
        for i, t in enumerate(tokens):
            if quote_mask[i]:
                continue
            if t not in {'def', 'async'}:
                continue
            if depths[i] != 0 and not probable_statement_boundary_at(i):
                continue
            parsed = parse_def(i)
            if parsed is not None:
                functions.append(parsed)
        if not functions:
            return []
        functions.sort(key=lambda x: (x[2], x[3], x[0], x[4]))
        return functions[0][1]

    def extract_if_structure(self, X):
        if not X:
            return []
        X = [str(t) for t in X]
        n = len(X)

        assign_ops = {
            "=", "+=", "-=", "*=", "/=", "//=", "%=",
            "&=", "|=", "^=", ":="
        }

        simple_starters = {
            "return", "pass", "break", "continue", "raise",
            "import", "from", "global", "nonlocal", "assert",
            "yield", "del"
        }

        open_to_close = {
            "(": ")",
            "[": "]",
            "{": "}"
        }

        close_to_open = {
            ")": "(",
            "]": "[",
            "}": "{"
        }

        def is_identifier(t):
            return (
                isinstance(t, str)
                and t != ""
                and (t[0].isalpha() or t[0] == "_")
                and all(c.isalnum() or c == "_" for c in t)
            )

        def is_newline_at(idx):
            return (
                idx < n
                and (
                    X[idx] in {"\n", "NEWLINE", "NL"}
                    or (
                        X[idx] == "\\"
                        and idx + 1 < n
                        and X[idx + 1] == "n"
                    )
                )
            )

        def newline_step(idx):
            if (
                idx < n
                and X[idx] == "\\"
                and idx + 1 < n
                and X[idx + 1] == "n"
            ):
                return 2
            return 1

        def is_space_token(t):
            return isinstance(t, str) and t.strip() == "" and t not in {"\\", "\n"}

        def is_indent_token(t):
            return t in {"INDENT", "<INDENT>"}

        def is_dedent_token(t):
            return t in {"DEDENT", "<DEDENT>"}

        def skip_layout(idx):
            while idx < n:
                if is_newline_at(idx):
                    idx += newline_step(idx)
                    while idx < n and is_space_token(X[idx]):
                        idx += 1
                elif is_space_token(X[idx]) or is_indent_token(X[idx]):
                    idx += 1
                else:
                    break
            return idx

        def skip_after_colon(idx):
            while idx < n:
                if is_newline_at(idx):
                    idx += newline_step(idx)
                    while idx < n and is_space_token(X[idx]):
                        idx += 1
                elif is_space_token(X[idx]):
                    idx += 1
                else:
                    break
            return idx

        def find_top_colon(start):
            stack = []
            k = start

            while k < n:
                t = X[k]
                if is_newline_at(k) and not stack:
                    return -1

                if t in open_to_close:
                    stack.append(t)
                elif t in close_to_open:
                    if stack and stack[-1] == close_to_open[t]:
                        stack.pop()
                elif not stack:
                    if t == ":":
                        return k
                    if t in assign_ops and t != ":=":
                        return -1
                    if (
                        k > start
                        and is_identifier(t)
                        and k + 1 < n
                        and X[k + 1] in assign_ops
                        and X[k + 1] != ":="
                    ):
                        return -1
                k += 1
            return -1

        def has_top_colon_before_boundary(start):
            stack = []
            k = start + 1
            while k < n:
                t = X[k]
                if t in open_to_close:
                    stack.append(t)
                elif t in close_to_open:
                    if stack and stack[-1] == close_to_open[t]:
                        stack.pop()
                elif not stack:
                    if t == ":":
                        return True                    
                    if is_newline_at(k) or t == ";":
                        return False
                    if t in assign_ops and t != ":=":
                        return False
                    if (
                        k > start + 1
                        and is_identifier(t)
                        and k + 1 < n
                        and X[k + 1] in assign_ops
                        and X[k + 1] != ":="
                    ):
                        return False
                k += 1
            return False

        def looks_like_statement_start(idx):
            if idx >= n:
                return False
            t = X[idx]
            if t in {"elif", "else", "except", "finally"}:
                return True
            if t in {"if", "for", "while", "with", "def", "class", "try"}:
                return has_top_colon_before_boundary(idx)
            if (
                t == "async"
                and idx + 1 < n
                and X[idx + 1] in {"def", "for", "with"}
            ):
                return has_top_colon_before_boundary(idx + 1)

            if t in simple_starters:
                return True
            
            if is_identifier(t) and idx + 1 < n and X[idx + 1] in assign_ops:
                return True

            return False

        def parse_simple_statement(start):
            if start < 0 or start >= n:
                return -1
            stack = []
            k = start
            seen = False
            start_tok = X[start]
            while k < n:
                t = X[k]
                if seen and not stack:
                    if is_newline_at(k):
                        return k + newline_step(k)
                    if t == ";":
                        return k + 1
                    # from os import path 
                    if not (start_tok == "from" and t == "import"):
                        if looks_like_statement_start(k):
                            return k
                if t in open_to_close:
                    stack.append(t)
                elif t in close_to_open:
                    if stack and stack[-1] == close_to_open[t]:
                        stack.pop()
                seen = True
                k += 1
            return k if seen else -1

        def parse_statement(start):
            start = skip_layout(start)
            if start < 0 or start >= n:
                return -1
            t = X[start]
            if is_dedent_token(t):
                return start

            if t == "if":
                return parse_if(start, include_chain=True)

            if t in {"for", "while", "with", "def", "class", "try"}:
                return parse_compound(start)

            if (
                t == "async"
                and start + 1 < n
                and X[start + 1] in {"def", "for", "with"}
            ):
                return parse_compound(start)
            return parse_simple_statement(start)

        def parse_block_after_colon(colon_idx):
            body_start = skip_after_colon(colon_idx + 1)
            if body_start >= n:
                return -1
            if body_start < n and is_indent_token(X[body_start]):
                while body_start < n and is_indent_token(X[body_start]):
                    body_start += 1
                k = body_start
                while k < n:
                    if is_dedent_token(X[k]):
                        return k + 1
                    if (
                        is_space_token(X[k])
                        or is_newline_at(k)
                        or is_indent_token(X[k])
                    ):
                        nxt = skip_layout(k)
                        k = nxt if nxt > k else k + 1
                        continue
                    end = parse_statement(k)
                    if end == -1 or end <= k:
                        k += 1
                    else:
                        k = end
                return k
            
            body_start = skip_layout(body_start)
            return parse_statement(body_start)

        def parse_compound(start):
            head_start = start
            if X[start] == "async":
                if start + 1 >= n or X[start + 1] not in {"def", "for", "with"}:
                    return -1
                head_start = start + 1
            colon = find_top_colon(head_start + 1)
            if colon == -1:
                return -1
            return parse_block_after_colon(colon)

        def parse_if(start, include_chain=True):
            if start < 0 or start >= n or X[start] != "if":
                return -1
            
            colon = find_top_colon(start + 1)
            if colon == -1:
                return -1
            
            end = parse_block_after_colon(colon)
            if end == -1 or end <= colon:
                return -1

            if not include_chain:
                return end
            #  elif / else
            k = skip_layout(end)
            while k < n:
                if X[k] == "elif":
                    colon2 = find_top_colon(k + 1)
                    if colon2 == -1:
                        break
                    end2 = parse_block_after_colon(colon2)
                    if end2 == -1 or end2 <= colon2:
                        break
                    end = end2
                    k = skip_layout(end)
                    continue
                if X[k] == "else":
                    if k + 1 >= n or X[k + 1] != ":":
                        break
                    end2 = parse_block_after_colon(k + 1)
                    if end2 == -1 or end2 <= k + 1:
                        break
                    end = end2
                    k = skip_layout(end)
                    continue
                break
            return end

        def logical_line_score(tokens):
            if not tokens:
                return 10 ** 9
            has_newline = any(t in {"\n", "NEWLINE", "NL"} for t in tokens)
            if has_newline:
                lines = 0
                has_content = False
                i = 0
                while i < len(tokens):
                    t = tokens[i]
                    if (
                        t in {"\n", "NEWLINE", "NL"}
                        or (
                            t == "\\"
                            and i + 1 < len(tokens)
                            and tokens[i + 1] == "n"
                        )
                    ):
                        if has_content:
                            lines += 1
                            has_content = False
                        if (
                            t == "\\"
                            and i + 1 < len(tokens)
                            and tokens[i + 1] == "n"
                        ):
                            i += 2
                        else:
                            i += 1
                        continue
                    if (
                        not is_space_token(t)
                        and not is_indent_token(t)
                        and not is_dedent_token(t)
                    ):
                        has_content = True
                    i += 1
                if has_content:
                    lines += 1
                return max(lines, 1)
            
            line_like = 0
            for t in tokens:
                if t in {
                    "if", "elif", "else", "for", "while", "with",
                    "def", "class", "try", "except", "finally"
                }:
                    line_like += 1
            return max(line_like, 1)
        candidates = []
        for i, tok in enumerate(X):
            if tok == "if":
                end = parse_if(i, include_chain=True)
                if end != -1 and end > i:
                    candidates.append(X[i:end])
        if not candidates:
            return []
        return min(candidates, key=lambda s: (logical_line_score(s), len(s)))

    def extract_loop_structure(self, X):
        if not X:
            return []
        n = len(X)
        ASSIGN_OPS = {
            '=', '+=', '-=', '*=', '/=', '//=', '%=', '**=',
            '&=', '|=', '^=', '>>=', '<<=', ':='
        }
        COMPOUND_START = {
            'if', 'for', 'while', 'try', 'with', 'def', 'class'
        }
        BRANCH_START = {
            'elif', 'else', 'except', 'finally'
        }
        SIMPLE_KEYWORDS = {
            'return', 'print', 'import', 'from', 'raise', 'assert',
            'break', 'continue', 'pass', 'del', 'global', 'nonlocal', 'yield'
        }
        OPEN = {
            '(': ')',
            '[': ']',
            '{': '}'
        }
        CLOSE = {
            ')': '(',
            ']': '[',
            '}': '{'
        }
        QUOTES = {'"', "'"}
        BLOCK_PREV = {
            '.', '=', '+=', '-=', '*=', '/=', '//=', '%=', '**=',
            '&=', '|=', '^=', '>>=', '<<=', ':=',
            '+', '-', '*', '/', '//', '%', '**',
            '<', '>', '==', '!=', '<=', '>=',
            'and', 'or', 'not', 'in', 'is', 'as',
            'raise', 'return', 'yield', 'assert',
            'from', 'import', 'del', 'lambda',
            ',', ':', '(', '[', '{'
        }
        def is_identifier(tok):
            return isinstance(tok, str) and tok.isidentifier()

        def skip_quote(i):
            q = X[i]
            j = i + 1
            while j < n:
                if X[j] == q:
                    return j + 1
                j += 1
            return i + 1

        def build_quote_mask_and_depths():
            quote_mask = [False] * n
            depths = [0] * n
            depth = 0
            i = 0
            while i < n:
                depths[i] = depth
                tok = X[i]
                if tok in QUOTES:
                    j = skip_quote(i)
                    for k in range(i, min(j, n)):
                        quote_mask[k] = True
                        depths[k] = depth
                    i = j
                    continue
                if tok in OPEN:
                    depth += 1
                elif tok in CLOSE:
                    depth = max(0, depth - 1)
                i += 1
            return quote_mask, depths
        
        quote_mask, depths = build_quote_mask_and_depths()
        
        #Scans tokens from lo
        def scan_top_level_token_before_colon(lo, target):
            depth = 0
            i = lo
            while i < n:
                tok = X[i]
                if tok in QUOTES:
                    i = skip_quote(i)
                    continue
                if tok in OPEN:
                    depth += 1
                elif tok in CLOSE:
                    depth = max(0, depth - 1)
                elif depth == 0:
                    if tok == target:
                        return True
                    if tok in (':', ';'):
                        return False
                i += 1
            return False

        def top_level_token_exists(lo, hi, token):
            depth = 0
            i = lo

            while i < hi:
                tok = X[i]

                if tok in QUOTES:
                    i = skip_quote(i)
                    continue
                if tok in OPEN:
                    depth += 1
                elif tok in CLOSE:
                    depth = max(0, depth - 1)
                elif tok == token and depth == 0:
                    return True
                i += 1
            return False

        def probable_statement_boundary_at(i):
            if i <= 0:
                return False
            prev = X[i - 1]
            return prev not in BLOCK_PREV

        def find_header_colon(start, owner_pos=None):
            depth = 0
            i = start
            ternary_if = False
            owner_tok = None
            if owner_pos is not None and 0 <= owner_pos < n:
                owner_tok = X[owner_pos]
            while i < n:
                tok = X[i]
                if tok in QUOTES:
                    i = skip_quote(i)
                    continue
                if tok in OPEN:
                    depth += 1
                elif tok in CLOSE:
                    depth = max(0, depth - 1)
                elif depth == 0:
                    if tok == ':':
                        return i
                    
                    if tok == ';':
                        return -1

                    if (
                        owner_tok in (COMPOUND_START | BRANCH_START)
                        and i > start
                        and tok in (COMPOUND_START | BRANCH_START)
                    ):
                        if tok == 'if' and scan_top_level_token_before_colon(i + 1, 'else'):
                            ternary_if = True
                        elif tok == 'else' and ternary_if:
                            ternary_if = False
                        elif probable_statement_boundary_at(i):
                            return -1
                i += 1
            return -1

        def header_valid(pos, colon):
            if colon == -1 or pos >= n:
                return False
            tok = X[pos]
            if tok == 'for':
                return top_level_token_exists(pos + 1, colon, 'in')
            if tok in ('while', 'if', 'with', 'def', 'class', 'elif', 'except'):
                return pos + 1 < colon
            if tok in ('try', 'else', 'finally'):
                return True
            return False

        def skip_balanced(pos):
            if pos >= n or X[pos] not in OPEN:
                return pos + 1
            open_tok = X[pos]
            close_tok = OPEN[open_tok]
            depth = 1
            i = pos + 1
            while i < n:
                tok = X[i]
                if tok in QUOTES:
                    i = skip_quote(i)
                    continue
                if tok == open_tok:
                    depth += 1
                elif tok == close_tok:
                    depth -= 1
                    if depth == 0:
                        return i + 1
                i += 1
            return n

        def assignment_op_after_lhs(pos):
            if pos >= n or quote_mask[pos] or not is_identifier(X[pos]):
                return -1            
            i = pos + 1
            while i < n:
                if X[i] in ASSIGN_OPS:
                    return i
                if X[i] == '.' and i + 1 < n and is_identifier(X[i + 1]):
                    i += 2
                    continue
                if X[i] == '[':
                    i = skip_balanced(i)
                    continue
                if X[i] == ',' and i + 1 < n and is_identifier(X[i + 1]):
                    i += 2
                    continue
                break
            return -1

        def looks_like_assignment_start(pos):
            return assignment_op_after_lhs(pos) != -1

        def looks_like_call_start(pos):
            if pos >= n or not is_identifier(X[pos]):
                return False
            i = pos + 1
            while i + 1 < n and X[i] == '.' and is_identifier(X[i + 1]):
                i += 2
            return i < n and X[i] == '('

        def looks_like_compound_start(pos):
            if (
                pos >= n
                or quote_mask[pos]
                or X[pos] not in (COMPOUND_START | BRANCH_START)
            ):
                return False
            colon = find_header_colon(pos + 1, pos)
            return header_valid(pos, colon)

        def raw_statement_start(pos):
            if pos >= n or quote_mask[pos]:
                return False
            tok = X[pos]
            if looks_like_compound_start(pos):
                return True

            if tok in SIMPLE_KEYWORDS:
                return True

            if looks_like_assignment_start(pos):
                return True

            if looks_like_call_start(pos):
                return True
            return False

        def split_allowed(pos, start):
            if pos <= start:
                return False           
            prev = X[pos - 1]
            tok = X[pos]
            if prev == '.':
                return False
            if X[start] == 'from' and tok == 'import':
                return False
            if prev in BLOCK_PREV:
                return False
            return True

        def parse_simple(pos):
            start = pos
            if pos >= n:
                return pos, 0
            if X[pos] in ('break', 'continue', 'pass'):
                return pos + 1, 1
            depth = 0
            i = pos
            while i < n:
                tok = X[i]

                if tok in QUOTES:
                    i = skip_quote(i)
                    continue
                if tok in OPEN:
                    depth += 1
                elif tok in CLOSE:
                    depth = max(0, depth - 1)
                elif tok == ';' and depth == 0:
                    return i + 1, 1
                if (
                    depth == 0
                    and i > pos
                    and raw_statement_start(i)
                    and split_allowed(i, start)
                ):
                    return i, 1
                i += 1
            return n, 1 if start < n else 0
        def parse_branch(start):
            colon = find_header_colon(start + 1, start)
            if not header_valid(start, colon):
                return start + 1, 1
            body_end, body_lines = parse_stmt(colon + 1)
            return body_end, 1 + body_lines

        def parse_compound(start):
            """
                if / for / while / try / with / def / class
            """
            tok = X[start]
            colon = find_header_colon(start + 1, start)
            if not header_valid(start, colon):
                return parse_simple(start)
            body_end, body_lines = parse_stmt(colon + 1)
            end = body_end
            lines = 1 + body_lines
            if tok == 'if':
                while (
                    end < n
                    and not quote_mask[end]
                    and X[end] in ('elif', 'else')
                    and looks_like_compound_start(end)
                ):
                    b_end, b_lines = parse_branch(end)
                    if b_end <= end:
                        break
                    end = b_end
                    lines += b_lines

                return end, lines

            if tok in ('for', 'while'):
                if (
                    end < n
                    and not quote_mask[end]
                    and X[end] == 'else'
                    and looks_like_compound_start(end)
                ):
                    b_end, b_lines = parse_branch(end)
                    if b_end > end:
                        end = b_end
                        lines += b_lines
                return end, lines
            if tok == 'try':
                while (
                    end < n
                    and not quote_mask[end]
                    and X[end] in ('except', 'else', 'finally')
                    and looks_like_compound_start(end)
                ):
                    b_end, b_lines = parse_branch(end)
                    if b_end <= end:
                        break
                    end = b_end
                    lines += b_lines

                return end, lines

            return end, lines

        def parse_stmt(pos):
            """
            Parse a statement
            """
            if pos >= n:
                return pos, 0

            if looks_like_compound_start(pos):
                return parse_compound(pos)
            
            return parse_simple(pos)
        loops = []
        for i, tok in enumerate(X):
            if quote_mask[i]:
                continue
            # depth == 0：
            if (
                tok in ('for', 'while')
                and depths[i] == 0
                and looks_like_compound_start(i)
            ):
                end, logical_lines = parse_compound(i)

                if end > i:
                    actual_start = i

                    # Support async for。
                    if (
                        tok == 'for'
                        and i > 0
                        and X[i - 1] == 'async'
                        and not quote_mask[i - 1]
                        and depths[i - 1] == 0
                    ):
                        actual_start = i - 1
                    loop_tokens = X[actual_start:end]
                    token_count = end - actual_start
                    loops.append(
                        (
                            logical_lines,
                            token_count,
                            actual_start,
                            loop_tokens
                        )
                    )
        if not loops:
            return []        
        loops.sort(key=lambda item: (item[0], item[1], item[2]))
        return loops[0][3]

    def extract_variable_statements(self, X):
        if not X:
            return []
        n = len(X)
        BAD_FOLLOW = {
            "+", "-", "*", "/", "//", "%", "**", "@",
            ".", "[", "(", "<", ">", "<=", ">=", "==", "!=",
            "and", "or", "in", "is"
        }
        STATEMENT_START = {
            "if", "elif", "else", "for", "while", "def", "class", "return",
            "import", "from", "try", "except", "finally", "with",
            "break", "continue", "pass", "print"
        }
        def is_identifier(t):
            return (
                isinstance(t, str)
                and t != ""
                and (t[0].isalpha() or t[0] == "_")
                and all(c.isalnum() or c == "_" for c in t)
            )
        def find_matching_paren(start):
            if start < 0 or start >= n or X[start] != "(":
                return -1
            depth = 0
            for k in range(start, n):
                if X[k] == "(":
                    depth += 1
                elif X[k] == ")":
                    depth -= 1
                    if depth == 0:
                        return k
            return -1
        def is_one_token_string(t):
            return (
                isinstance(t, str)
                and len(t) >= 2
                and (
                    (t[0] == '"' and t[-1] == '"')
                    or (t[0] == "'" and t[-1] == "'")
                )
            )
        def parse_string_literal(pos):
            # ['s', '=', '"abc"']
            if pos < n and is_one_token_string(X[pos]):
                return pos + 1
            #['s', '=', '"', 'abc', '"']
            if pos < n and X[pos] in {'"', "'"}:
                quote = X[pos]
                k = pos + 1
                while k < n:
                    if X[k] == quote:
                        return k + 1
                    k += 1
            return -1

        def is_number_token(t):
            return (
                isinstance(t, str)
                and re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)', t) is not None
            )

        def parse_number_literal(pos):
            if pos >= n:
                return -1

            # ['-', '3', '.', '14']
            if (
                pos + 3 < n
                and X[pos] in {"+", "-"}
                and isinstance(X[pos + 1], str) and X[pos + 1].isdigit()
                and X[pos + 2] == "."
                and isinstance(X[pos + 3], str) and X[pos + 3].isdigit()
            ):
                return pos + 4

            # ：['3', '.', '14']
            if (
                pos + 2 < n
                and isinstance(X[pos], str) and X[pos].isdigit()
                and X[pos + 1] == "."
                and isinstance(X[pos + 2], str) and X[pos + 2].isdigit()
            ):
                return pos + 3

            # ：['-', '1'] / ['+', '1']
            if (
                pos + 1 < n
                and X[pos] in {"+", "-"}
                and is_number_token(X[pos + 1])
                and not str(X[pos + 1]).startswith(("+", "-"))
            ):
                return pos + 2

            # ：['1'] / ['-1'] / ['3.14']
            if is_number_token(X[pos]):
                return pos + 1
            return -1

        def parse_empty_container(pos):
            # ：['arr', '=', '[]']
            if pos < n and X[pos] in {"[]", "{}", "()"}:
                return pos + 1

            # ：['arr', '=', '[', ']']
            pairs = {"[": "]", "{": "}", "(": ")"}
            if pos + 1 < n and X[pos] in pairs and X[pos + 1] == pairs[X[pos]]:
                return pos + 2

            return -1

        def parse_literal(pos):
            if pos >= n:
                return -1
            if X[pos] in {"True", "False", "true", "false"}:
                return pos + 1
            for parser in (parse_number_literal, parse_string_literal, parse_empty_container):
                end = parser(pos)
                if end != -1:
                    return end
            return -1

        def starts_extractable_statement(pos):
            if pos >= n:
                return True
            if X[pos] in {"\n", ";"}:
                return True
            if X[pos] in STATEMENT_START:
                return True
            if X[pos] == "print" and pos + 1 < n and X[pos + 1] == "(":
                return True
            if (
                is_identifier(X[pos])
                and pos + 2 < n
                and X[pos + 1] == "="
                and (
                    X[pos + 2] in {"input", "raw_input"}
                    or parse_literal(pos + 2) != -1
                )
            ):
                return True
            return False

        def clean_statement_end(end):
            if end < n and X[end] in BAD_FOLLOW:
                return False
            return starts_extractable_statement(end)

        def match_print(pos):
            if pos + 1 < n and X[pos] == "print" and X[pos + 1] == "(":
                close = find_matching_paren(pos + 1)
                end = close + 1
                if close != -1 and clean_statement_end(end):
                    return end
            return -1

        def match_input_assignment(pos):
            if not (
                is_identifier(X[pos])
                and pos + 4 < n
                and X[pos + 1] == "="
                and X[pos + 2] in {"input", "raw_input"}
                and X[pos + 3] == "("
            ):
                return -1

            close = find_matching_paren(pos + 3)
            end = close + 1
            if close != -1 and clean_statement_end(end):
                return end
            return -1

        def match_simple_assignment(pos):
            if not (
                is_identifier(X[pos])
                and pos + 2 < n
                and X[pos + 1] == "="
            ):
                return -1
            end = parse_literal(pos + 2)
            if end != -1 and clean_statement_end(end):
                return end
            return -1
        result = []
        i = 0
        while i < n:
            end = match_print(i)
            if end != -1:
                result.extend(X[i:end])
                i = end
                continue
            end = match_input_assignment(i)
            if end != -1:
                result.extend(X[i:end])
                i = end
                continue
            end = match_simple_assignment(i)
            if end != -1:
                result.extend(X[i:end])
                i = end
                continue
            i += 1
        return result


    def wrap_block(self, inst, context_tokens=None, guard_name="check_flag_0"):
        """
        inst = self.wrap_block(inst, context_tokens=tok)
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

        def pick_natural_guard_name(tokens):
            preferred_names = [
                "debug", "verbose", "enabled", "flag", "valid", "ready",
                "active", "allow", "ok", "done", "found", "skip",
                "use_cache", "need_check", "is_valid", "should_skip",
                "check_flag", "check_flag_0"
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
                    if t in {"case", "cases", "data", "line", "temp", "num", "nums"}:
                        continue
                    if nxt == "=" and rhs in {"0", "1", "False", "True", "false", "true"}:
                        candidates.append(t)
                if candidates:
                    candidates = list(set(candidates))
                    return random.choice(candidates)
            return random.choice([
                "debug", "enabled", "valid", "ready",
                "active", "check_flag", "check_flag_0",
                "need_check", "should_skip"
            ])

        def add_outer_indent(tokens, indent="    "):
            if not tokens:
                return []
            result = []
            at_line_start = True

            for idx, tok in enumerate(tokens):
                if at_line_start:
                    result.append(indent)
                    at_line_start = False
                result.append(tok)
                if tok == "\n":
                    at_line_start = True
                    if idx == len(tokens) - 1:
                        at_line_start = False
            return result
        guard_name = pick_natural_guard_name(context_tokens)
        false_guard_templates = [
            ["if", "0", "and", guard_name, ":", "\n"],
            ["if", "False", "and", guard_name, ":", "\n"],
            ["if", "not", "True", "and", guard_name, ":", "\n"],
            ["if", "1", "<", "0", "and", guard_name, ":", "\n"],
            ["if", "(", "0", ")", "and", guard_name, ":", "\n"],
            ["if", "0", "and", "not", guard_name, ":", "\n"],
            ["if", "False", "and", "not", guard_name, ":", "\n"],
        ]
        prefix = random.choice(false_guard_templates)
        return prefix + add_outer_indent(inst)

    def trim_inst_by_complete_stmt(self, inst, max_tokens=50):
        if not inst:
            return []
        if len(inst) <= max_tokens:
            return inst
        assign_ops = {
            "=", "+=", "-=", "*=", "/=", "//=", "%=", "**=",
            "&=", "|=", "^=", ">>=", "<<=", ":="
        }
        stmt_starters = {
            "if", "for", "while", "with", "try", "def", "class",
            "return", "print", "raise", "assert", "yield", "del",
            "break", "continue", "pass", "import", "from",
            "global", "nonlocal"
        }
        open_to_close = {
            "(": ")",
            "[": "]",
            "{": "}"
        }
        close_to_open = {
            ")": "(",
            "]": "[",
            "}": "{"
        }
        block_prev = {
            ".", "=", "+=", "-=", "*=", "/=", "//=", "%=", "**=",
            "&=", "|=", "^=", ">>=", "<<=", ":=",
            "+", "-", "*", "/", "//", "%", "**",
            "<", ">", "<=", ">=", "==", "!=",
            "and", "or", "not", "in", "is", "as",
            "return", "yield", "raise", "assert",
            "from", "import", "lambda",
            ",", ":", "(", "[", "{", "->"
        }
        def is_identifier(t):
            return (
                isinstance(t, str)
                and t != ""
                and (t[0].isalpha() or t[0] == "_")
                and all(c.isalnum() or c == "_" for c in t)
            )
        def looks_like_stmt_start(i):
            if i >= len(inst):
                return False
            t = inst[i]
            if t in stmt_starters:
                return True
            if is_identifier(t) and i + 1 < len(inst):
                if inst[i + 1] in assign_ops:
                    return True
                if inst[i + 1] == "(":
                    return True
            return False

        def split_allowed(i):
            if i <= 0:
                return False
            prev = inst[i - 1]
            if prev in block_prev:
                return False
            return True
        stack = []
        safe_cuts = []
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
            if not stack:
                if t in {";", "\n", "NEWLINE", "NL"}:
                    safe_cuts.append(i + 1)
                elif i > 0 and looks_like_stmt_start(i) and split_allowed(i):
                    safe_cuts.append(i)
            i += 1
        valid_cuts = [c for c in safe_cuts if c <= max_tokens]
        if valid_cuts:
            return inst[:max(valid_cuts)]
        return []
    
    def insert4(self,x_token, x,rand_sa, n_candidate=5):
        pos_candidates = patternpy.InsAddCandidates(self.insertDict, self.cl.max_len) # exclude outlier poses
        n = len(pos_candidates)
        tok=x_token[0] 
        # numb=50
        numb=70  
        # print("x==============",x)
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
                    # print("inst=======",inst)
                    if inst:
                       if len(inst) > numb:
                            inst = self.trim_inst_by_complete_stmt(inst, max_tokens=numb)
                            if not inst:
                                continue                        
                       inst_idxs = self._insert2idxs(inst)
                    #    print("inst_idxs=======",inst_idxs)
                       _insertDict = deepcopy(self.insertDict)
                       suc=patternpy.InsAdd(_insertDict, pos, inst_idxs)
                       if not suc:
                          continue
                    #    print("pos:", pos, "=>", inst, "count", _insertDict["count"])
                       _x = patternpy.InsResult1(x,pos, _insertDict)
                       new_x.append(_x)
                       new_insertDict.append(_insertDict)
                    #    print("x=======",x)
                       return new_x, new_insertDict
                    else:  
                        if 'function' in fragment_types:
                            fragment_types.remove('function')
                        if not fragment_types:  
                            break
                elif selected_type == 'comment':
                    inst = self.extract_variable_statements(rand_d1)  
                    # print("'commen=",inst)
                    if inst:
                        break
                elif selected_type == 'selection':
                    inst = self.extract_if_structure(rand_d1)  
                    # print("if=",inst)
                    if inst:
                        break
                elif selected_type == 'loop':
                    inst = self.extract_loop_structure(rand_d1)  
                    # print("for=",inst)
                    if inst:
                        break
            # if not inst:        
            #     print("inst:////", inst)    
            _insertDict = deepcopy(self.insertDict)
            inst = self.wrap_block(inst, context_tokens=tok)
            if len(inst) > numb:
                inst = self.trim_inst_by_complete_stmt(inst, max_tokens=numb)
                if not inst:
                    continue            
            inst_idxs = self._insert2idxs(inst)           
            suc=patternpy.InsAdd(_insertDict, pos, inst_idxs)
            if not suc:
                continue
            #print("pos:", pos, "=>", inst, "count", _insertDict["count"])
            _x = patternpy.InsResult(x, _insertDict)
            new_x.append(_x)           
            new_insertDict.append(_insertDict)
        return new_x, new_insertDict
 
    def remove(self, x, n_candidate=5):
        pos_candidates = patternpy.InsDeleteCandidates(self.insertDict) # e.g. [(pos0, 0), (pos0, 1), (pos1, 0), ...]
        n = len(pos_candidates)
        if n_candidate < n:
          candisIdx = random.sample(range(n), n_candidate)
        else:
          candisIdx = random.sample(range(n), n)
        pos_candidates = [pos_candidates[candiIds] for candiIds in candisIdx]

        new_x, new_insertDict = [], [] 
        for pos, inPosIdx in pos_candidates:
            _insertDict = deepcopy(self.insertDict)
            patternpy.InsDelete(_insertDict, pos, inPosIdx)
            _x = patternpy.InsResult(x, _insertDict)
            new_x.append(_x)
            new_insertDict.append(_insertDict)

        return new_x, new_insertDict

    def insert_remove_random(self, x):

        new_x, new_insertDict = [], []
        fail_cnt = 0
        while True:
            if fail_cnt >= 10:  # in case of dead loop
                break
            if random.random() > 0.5: # insert
                pos_candidates = patternpy.InsAddCandidates(self.insertDict, self.cl.max_len) # exclude outlier poses
                if pos_candidates == []:
                    fail_cnt += 1
                    continue
                pos_cand = random.sample(pos_candidates, 1)[0]
                inst = random.sample(self.inserts, 1)[0]
                inst_idxs = self._insert2idxs(inst)
                _insertDict = deepcopy(self.insertDict)
                patternpy.InsAdd(_insertDict, pos_cand, inst_idxs)
            else:
                pos_candidates = patternpy.InsDeleteCandidates(self.insertDict)
                if pos_candidates == []:
                    fail_cnt += 1
                    continue
                pos_cand, inPosIdx = random.sample(pos_candidates, 1)[0]
                _insertDict = deepcopy(self.insertDict)
                patternpy.InsDelete(_insertDict, pos_cand, inPosIdx)
            _x = patternpy.InsResult(x, _insertDict)
            new_x.append(_x)
            new_insertDict.append(_insertDict)
            break
        return new_x, new_insertDict

def idxs2tokens(idxs, idx2word, unk_idx):
    res = []
    n = len(idx2word)
    for idx in idxs:
      if idx < n:
        res.append(idx2word[idx])
      else:
        res.append(idx2word[unk_idx])
    return res

if __name__ == "__main__":
    
    from dataset import Author66
    from lstm_classifier import LSTMClassifier, LSTMEncoder
    import argparse
    import pickle, gzip, os, sys
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default="0")
    parser.add_argument('-attn', action='store_true')
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    device = torch.device("cuda")
    
    vocab_size = 5000
    embedding_size = 512
    hidden_size = 600
    n_layers = 2
    num_classes = 66
    max_len = 600

    poj = Author66(path="../data_author/author.pkl.gz", max_len=max_len, vocab_size=vocab_size)
    training_set, valid_set, test_set = poj.train, poj.dev, poj.test
    with gzip.open('../data_author/author_uid.pkl.gz', "rb") as f:
        symtab = pickle.load(f)
    with gzip.open('../data_author/author_inspos.pkl.gz', "rb") as f:
        instab = pickle.load(f)
    
    enc = LSTMEncoder(embedding_size, hidden_size, n_layers)
    classifier = LSTMClassifier(vocab_size, embedding_size, enc,
                                hidden_size, num_classes, max_len, attn=opt.attn).cuda()
    classifier.load_state_dict(torch.load('/../MODEL/PATH12.pt'))
    
    b = test_set.next_batch(1)
    stmt_ins_poses = instab['stmt_te'][b['id'][0]]
    m = InsModifier(classifier, poj.get_txt2idx(), stmt_ins_poses)

    patternpy._InsVis(b['raw'][0], stmt_ins_poses)

    x = torch.tensor(b['x'], dtype=torch.long).cuda().permute([1, 0])
    y = torch.tensor(b['y'], dtype=torch.long).cuda()
    print (b['y'][0])
    prob = classifier.prob(x)[0]
    print (int(torch.argmax(prob)), float(prob[b['y'][0]]))
    
    old_x = b['x'][0]
    for _ in range(2):
        new_x, new_insertDict = m.insert(old_x, n_candidate=3)
        feed_new_x = [_x[:classifier.max_len] for _x in new_x]  # this step is very important
        new_prob = classifier.prob(torch.tensor(feed_new_x, dtype=torch.long).cuda().permute([1, 0]))
        for _p, _dict in zip(new_prob, new_insertDict):
            print (float(_p[b['y'][0]]), int(torch.argmax(_p).cpu()))
        m.insertDict = new_insertDict[0]
        print ('------------ INSERT -------------', m.insertDict["count"])
    for _ in range(2):
        new_x, new_insertDict = m.remove(old_x, n_candidate=3)
        feed_new_x = [_x[:classifier.max_len] for _x in new_x]  # this step is very important
        new_prob = classifier.prob(torch.tensor(feed_new_x, dtype=torch.long).cuda().permute([1, 0]))
        for _p, _dict in zip(new_prob, new_insertDict):
            print (float(_p[b['y'][0]]), int(torch.argmax(_p).cpu()))
        m.insertDict = new_insertDict[0]
        print ('------------ REMOVE -------------', m.insertDict["count"])
