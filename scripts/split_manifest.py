import argparse, os, pandas as pd
from sklearn.model_selection import train_test_split
p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
df=pd.read_csv(a.input); os.makedirs(a.output_dir,exist_ok=True)
# First split by design: no test design is seen during training.
design_counts=df.groupby('design_id').size()
eligible=design_counts[design_counts>=2].index
ineligible=design_counts[design_counts<2].index
if len(ineligible): print(f'WARNING: {len(ineligible)} designs have <2 images and cannot support gallery/query testing.')
designs=eligible.to_numpy()
tr, tmp=train_test_split(designs,test_size=0.30,random_state=a.seed)
va, te=train_test_split(tmp,test_size=0.50,random_state=a.seed)
df[df.design_id.isin(tr)].to_csv(os.path.join(a.output_dir,'train.csv'),index=False)
df[df.design_id.isin(va)].to_csv(os.path.join(a.output_dir,'val.csv'),index=False)
test=df[df.design_id.isin(te)].copy()
# Within each test design, put at least one image in gallery and one in query.
gallery_parts=[]; query_parts=[]
for did,g in test.groupby('design_id'):
    g=g.sample(frac=1,random_state=a.seed)
    n=max(1,len(g)//2)
    gallery_parts.append(g.iloc[:n]); query_parts.append(g.iloc[n:])
gallery=pd.concat(gallery_parts).reset_index(drop=True)
query=pd.concat(query_parts).reset_index(drop=True)
gallery.to_csv(os.path.join(a.output_dir,'gallery.csv'),index=False)
query.to_csv(os.path.join(a.output_dir,'query.csv'),index=False)
# Keep a combined test file for pair-level verification.
test.to_csv(os.path.join(a.output_dir,'test.csv'),index=False)
print('Train designs:',len(tr),'Val designs:',len(va),'Test designs:',len(te))
print('Gallery images:',len(gallery),'Query images:',len(query),'Test images:',len(test))
