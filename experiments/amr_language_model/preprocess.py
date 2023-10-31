import os

import numpy as np
import penman as pm
from penman.transform import canonicalize_roles
from penman import surface
import numpy

if __name__ == '__main__':
    data_dir = r'C:\Users\lu\Desktop\amr_annotation_3.0\data\alignments\unsplit'
    from os import walk
    from tqdm import tqdm
    filenames = next(walk(data_dir), (None, None, []))[2]
    all_entries = []
    for filename in filenames:
        with open(os.path.join(data_dir,filename), encoding="utf8") as f:
            a = f.read()
            a= a.split('\n\n')[1:]
            a
        def trace_to_end(instance,lookup,edges):

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

        def to_entry(text):
            try:
                gra = pm.decode(text)
                alignments = surface.alignments(gra)
                lookup = {k:v.indices for (k,_,_),v in alignments.items()}
                adjacency = [[ 'O' for i in range(len(gra.metadata['tok'].split()))] for i in range(len(gra.metadata['tok'].split()))]
                test = []
                for num,e in enumerate(gra.instances()):
                    sources = trace_to_end(e.source,lookup,gra.edges())
                    for s in sources:
                        for t in sources:
                            adjacency[s][t] = 'INSTANCE'+str(num)
                for e in gra.edges():
                    sources = trace_to_end(e.source,lookup,gra.edges())
                    targets = trace_to_end(e.target,lookup,gra.edges())
                    for s in sources:
                        for t in targets:
                            adjacency[s][t] = str(e.role)
                return {
                    'sent': gra.metadata['tok'],
                    'tok': gra.metadata['tok'].split(),
                    'adj':adjacency
                }
            except:
                print(text)

        all_entries.extend( [ to_entry(aa) for aa in tqdm(a)])
    all_entries = list(filter(lambda x:x is not None,all_entries))
    '''for e in all_entries:
        adj = e['adj']
        sss = set()
        for x,xx in enumerate(adj):
            for y,yy in enumerate(xx):
                    if not( x  == y or yy  == 'O'):
                        sss.add(x)
                        sss.add(y)
        for ss in sss:
            if adj[ss] == 'O':
                print('fuck: '+e['sent'])'''
    sss = set()
    for e in all_entries:
        adj = e['adj']
        for x, xx in enumerate(adj):
            for y, yy in enumerate(xx):
                if not ( yy == 'O'):
                    sss.add(adj[x][y])
    with open('roles.txt', 'w') as f:
        print(sss,file=f)

    import json
    with open('data.jsonl', 'w') as f:
        for e in tqdm(all_entries):
            final = json.dumps(e)
            f.write(final+'\n')