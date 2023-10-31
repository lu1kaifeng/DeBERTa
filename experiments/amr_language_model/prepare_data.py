# coding: utf-8
from DeBERTa import deberta
import sys
import argparse
from tqdm import tqdm
import numpy as np
data_jsonl = './data.jsonl'
if __name__ == '__main__':
  p,t=deberta.load_vocab(vocab_path=None, vocab_type='spm', pretrained_id='large')
  tokenizer=deberta.tokenizers[t](p)
  all_tokens = []
  import json
  adj_list = []
  token_list = []
  with open(data_jsonl, encoding = 'utf-8') as fs:
    for ll in fs:
      entry = json.loads(ll)
      tokens,traced = tokenizer.traced_tokenize(entry['sent'])
      adj = [ ['O' for ii in range(len(traced)) ] for i in range(len(traced))]
      for tx in range(len(traced)):
        for ty in range(len(traced)):
          adj[tx][ty] = entry['adj'][traced[tx]][traced[ty]]
      adj_list.append(adj)
      token_list.append(tokens)

  with open('adjs.txt','w', encoding='utf-8') as f:
        for adj in adj_list:
          print(adj,file=f)
  print('max context length: '+ str(len(max(token_list,key = lambda x:len(x)))))
  with open('toks.txt','w', encoding='utf-8') as f:
        for adj in token_list:
          print(' '.join(adj),file=f)

