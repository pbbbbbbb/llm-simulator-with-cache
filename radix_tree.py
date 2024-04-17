import tiktoken

class TreeNode:
    def __init__(self, token=None, keys=None, vals=None, child={}) -> None:
        self.token = token
        self.keys = keys
        self.vals = vals
        self.child = child

class RadixTree:
    def __init__(self, input_list=[], enc=None) -> None:
        self.enc = enc if enc != None else tiktoken.get_encoding("cl100k_base")
        self.build_tree(input_list)

    def build_tree(self, input_list):
        self.tree = TreeNode()
        for s in input_list:
            self.insert(s)

    def get(self, tokens):
        curr = self.tree
        for token in tokens:
            if token not in curr.child:
                return curr.keys, curr.vals
            curr = curr.child[token]
        return curr.keys, curr.vals

    def insert(self, tokens, keys=None, vals=None):
        curr = self.tree
        for token in tokens:
            if token not in curr.child:
                curr.child[token] = TreeNode(token=token)
            curr = curr.child[token]
        curr.keys = keys
        curr.vals = vals

            