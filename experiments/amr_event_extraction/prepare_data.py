# coding: utf-8
from DeBERTa import deberta
import sys
import argparse
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    p, t = deberta.load_vocab(vocab_path=None, vocab_type='spm', pretrained_id='large')
    tokenizer = deberta.tokenizers[t](p)
    import json
    import numpy as np
    import penman as pm
    from penman.transform import canonicalize_roles
    from penman import surface
    import numpy


    def trace_to_end(instance, lookup, edges):

        def _walk(source):
            candidate = []
            for e in edges:
                if e.source == source:
                    candidate.append(e)
            indices = []
            for c in candidate:
                if c.target in lookup:
                    indices.extend(lookup[c.target])
                else:
                    indices.extend(_walk(c.target))
            return indices

        if instance not in lookup:
            return tuple(set(_walk(instance)))
        else:
            return lookup[instance]


    count = 0


    def to_entry(text, tok):
        adjacency = [['O' for i in range(len(tok))] for i in
                     range(len(tok))]
        if text is None:
            return adjacency
        try:
            gra = pm.decode(text)
            alignments = surface.alignments(gra)
            lookup = {k: v.indices for (k, _, _), v in alignments.items()}

            test = []
            for num, e in enumerate(gra.instances()):
                sources = trace_to_end(e.source, lookup, gra.edges())
                for s in sources:
                    for t in sources:
                        adjacency[s][t] = 'INSTANCE' + str(num)
            for e in gra.edges():
                sources = trace_to_end(e.source, lookup, gra.edges())
                targets = trace_to_end(e.target, lookup, gra.edges())
                for s in sources:
                    for t in targets:
                        adjacency[s][t] = str(e.role)
            return adjacency

        except Exception as jk:
            print(jk)
            print(tok)
            print(text)
            global count
            count += 1
            return adjacency

for i, o in zip(['train_ali.json', 'dev_ali.json', 'test_ali.json'],
                ['train_mat.json', 'dev_mat.json', 'test_mat.json']):
    with open(i, encoding='utf-8') as fs:
        entry = json.loads(fs.read())
        for e in entry:
            e['amr'] = to_entry(e['amrAlign'][0] if 'amrAlign' in e else None, e['words'])
            tokens, traced = tokenizer.traced_tokenize(e['sentence'])
            adj = [['O' for ii in range(len(traced))] for i in range(len(traced))]
            for tx in range(len(traced)):
                for ty in range(len(traced)):
                    adj[tx][ty] = e['amr'][traced[tx]][traced[ty]]
            e['amr'] = adj
        with open(o, 'w', encoding='utf-8') as fo:
            fo.write(json.dumps(entry))
print('Defective entries: '+str(count))
