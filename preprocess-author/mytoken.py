# -*- coding: utf-8 -*-
import os, sys
import shutil
import tarfile
import pickle
import tqdm
import re
from tree_sitter import Language, Parser
import io
import tokenize as py_tokenize

# tree-sitter
PY_LANGUAGE = Language('../parser_folder/my-languages.so', 'python')
parser = Parser()
parser.set_language(PY_LANGUAGE)

def unzip(dir='../preprocess-Author_data/tmp', done_file="unzip.done"):
    return os.path.isdir(os.path.join(dir, "ProgramData"))

def remove_comment(text):
    def replacer(match):
        s = match.group(0)
        if s.startswith('#'):
            return " "
        elif s.startswith('"""') or s.startswith("'''"):
            return " "
        else:
            return s
    pattern = re.compile(
        r'#.*?$|""".*?"""|\'\'\'.*?\'\'\'',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)


def python_tokenize_bytes(code_bytes):
    tokens = []
    try:
        readline = io.BytesIO(code_bytes).readline
        for tok in py_tokenize.tokenize(readline):
            if tok.type in (py_tokenize.ENCODING, py_tokenize.NL,
                            py_tokenize.NEWLINE, py_tokenize.INDENT, py_tokenize.DEDENT):
                continue
            s = tok.string
            if not s:
                continue
            s = s.strip()
            if s:
                tokens.append(s)
    except Exception:
        txt = code_bytes.decode('utf-8', errors='ignore')
        tokens = re.findall(r'\w+|[^\s\w]', txt)
    return tokens


def extract_identifiers(tree_or_none, code_bytes):
    ids = set()
    try:
        if tree_or_none is not None and hasattr(tree_or_none, "root_node"):
            root = tree_or_none.root_node

            def visit(node):
                # tree-sitter-python node： 'identifier', 'function_definition', 'class_definition'
                if node.type == "identifier":
                    ids.add(code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore"))
                elif node.type == "function_definition":
                    name = node.child_by_field_name("name")
                    if name:
                        ids.add(code_bytes[name.start_byte:name.end_byte].decode("utf-8", errors="ignore"))
                elif node.type == "class_definition":
                    name = node.child_by_field_name("name")
                    if name:
                        ids.add(code_bytes[name.start_byte:name.end_byte].decode("utf-8", errors="ignore"))
                for c in node.children:
                    visit(c)

            visit(root)
            return ids
    except Exception:
        pass
    try:
        import ast
        src = code_bytes.decode("utf-8", errors="ignore")
        class _V(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                ids.add(node.name)
                for a in node.args.args:
                    if isinstance(a.arg, str):
                        ids.add(a.arg)
                self.generic_visit(node)
            def visit_ClassDef(self, node):
                ids.add(node.name)
                self.generic_visit(node)
            def visit_Name(self, node):
                if isinstance(node.id, str):
                    ids.add(node.id)
                self.generic_visit(node)
            def visit_arg(self, node):
                if hasattr(node, "arg") and isinstance(node.arg, str):
                    ids.add(node.arg)
                self.generic_visit(node)
        tree_ast = ast.parse(src)
        _V().visit(tree_ast)
    except Exception:
        txt = code_bytes.decode("utf-8", errors="ignore")
        for name in re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', txt):
            ids.add(name)
    return ids

def tokenize(dir='/home/mabaoying/CARROT-main/preprocess-Author_data/tmp',
             src='ProgramData', tgt='tokenized.pkl',
             done_file="token.done", min_len=5, debug=False):
    if os.path.isdir(dir) and os.path.isfile(os.path.join(dir, done_file)):
        with open(os.path.join(dir, tgt), "rb") as f:
            return pickle.load(f)
    try:
        data = {'raw': [], "labels": [], "uids": []}
        label_to_index = {}
        base_src = os.path.join(dir, src)
        for label_idx, label in enumerate(sorted(os.listdir(base_src))):
            label_path = os.path.join(base_src, label)
            if not os.path.isdir(label_path):
                continue
            label_to_index[label] = label_idx
            for file in sorted(os.listdir(label_path)):
                if not file.endswith(".py"):
                    continue
                file_path = os.path.join(label_path, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    try:
                        text = remove_comment(text)
                    except Exception:
                        pass
                    text = text.replace('\r\n', '\n')
                    code_bytes = text.encode('utf-8')
                    tokens = python_tokenize_bytes(code_bytes)
                    tree = None
                    try:
                        parser  # noqa: F401
                        tree = parser.parse(code_bytes)
                    except Exception:
                        tree = None
                    idents = extract_identifiers(tree, code_bytes)
                    uids = {}
                    for i, tok in enumerate(tokens):
                        if tok in idents:
                            uids.setdefault(tok, []).append(i)
                    if debug:
                        print("File:", file_path)
                        print("  token count:", len(tokens))
                        print("  first tokens:", tokens[:50])
                        if tokens and ("\n" in tokens[0] or len(tokens[0]) > 200):
                            print("  WARN: first token looks large (repr):", repr(tokens[0]))
                    if len(tokens) >= min_len:
                        data['labels'].append(label_idx)
                        data['raw'].append(tokens)
                        data['uids'].append(uids)

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue
        with open(os.path.join(dir, tgt), "wb") as f:
            pickle.dump(data, f)
        with open(os.path.join(dir, done_file), "wb") as f:
            pass
        return data
    except Exception as e:
        print(f"Fatal error: {e}")
        return None
    
if __name__ == "__main__":
    if unzip():
        d = tokenize()