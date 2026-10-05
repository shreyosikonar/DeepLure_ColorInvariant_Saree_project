import argparse, os, pandas as pd
from PIL import Image
EXT={'.jpg','.jpeg','.png','.webp','.bmp'}
p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output',required=True); a=p.parse_args()
rows=[]; labels={}
for root,_,files in os.walk(a.input):
    design=os.path.basename(root)
    if design not in labels: labels[design]=len(labels)
    for f in files:
        path=os.path.join(root,f)
        if os.path.splitext(f)[1].lower() in EXT:
            try:
                with Image.open(path) as im: im.verify()
                rows.append({'path':os.path.abspath(path),'design_id':design,'label':labels[design]})
            except Exception: pass
pd.DataFrame(rows).to_csv(a.output,index=False)
print(f'Wrote {len(rows)} images across {len(labels)} design IDs to {a.output}')
