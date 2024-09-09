import tqdm
from amrlib.alignments.faa_aligner import FAA_Aligner
import json
inference = FAA_Aligner()
#only works in Linux
for i,o in zip(['dev.json','test.json','train.json'],['dev_ali.json','test_ali.json','train_ali.json']):
  with open(i, encoding = 'utf-8') as fs:
      entry = json.loads(fs.read())
      for e in tqdm.tqdm(entry):
        try:
            amr_surface_aligns, alignment_strings = inference.align_sents([e['sentence']], [e['amr']])
            e['amrAlign'] = amr_surface_aligns
        except Exception:
            print(e['sentence'], e['amr'])
            print('\n')

      with open(o, 'w', encoding='utf-8') as fo:
        print('writing '+o+'\n')
        fo.write(json.dumps(entry))
