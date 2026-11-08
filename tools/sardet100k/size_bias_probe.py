"""Zero-training probe: how small Mosaic makes targets at the 640 input, and whether E000 scores follow target size at equal pixel error.

Usage: python tools/sardet100k/size_bias_probe.py  (paths are the new-server ones used in docs/sardet100k_mosaic_seam_audit.md)
"""
import json, csv, numpy as np
from collections import defaultdict
A='/root/zxy/experiments/diagnostics/sardet100k_mosaic_seam_audit_20261108'
S={s['k']:s for s in json.load(open(A+'/samples.json'))}
rows=list(csv.DictReader(open(A+'/targets.csv')))
fin=np.array([float(r['long_side'])*640/max(S[int(r['sample'])]['size']) for r in rows]); cls=np.array([r['class'] for r in rows])
tr=json.load(open('/root/zxy/TCRA-SAR/datasets/SARDet_100K/Annotations/train.json'))
names={c['id']:c['name'] for c in tr['categories']}; size={i['id']:max(i['width'],i['height']) for i in tr['images']}
# no-mosaic path: Resize to 640x640 directly
nm=np.array([max(a['bbox'][2],a['bbox'][3])*640/size[a['image_id']] for a in tr['annotations']]); nc=np.array([names[a['category_id']] for a in tr['annotations']])
print('long side at network input (640): share <4 / <8 / <16')
for c in ['all','tank','bridge','ship','aircraft']:
    m=fin if c=='all' else fin[cls==c]; n=nm if c=='all' else nm[nc==c]
    f=lambda v:' '.join(f'{(v<t).mean()*100:5.1f}%' for t in (4,8,16))
    print(f'{c:9s} no-mosaic {f(n)} | mosaic {f(m)} | median {np.median(n):.1f} -> {np.median(m):.1f}')
# E000: TP score vs size at matched IoU
R='/root/zxy/experiments/deim_m_sardet100k_deim_hgnetv2_m_sardet100k_36e_b32_nozoom_nocrop_seed0_20261105_125147_N2zAJh/independent_verify_20261106/val_predictions.npz'
d=np.load(R); PI,PC,PS,PB=d['image_id'],d['label'].astype(int),d['score'],d['box']
val=json.load(open('/root/zxy/TCRA-SAR/datasets/SARDet_100K/Annotations/val.json'))
g=defaultdict(list)
for a in val['annotations']:
    x,y,w,h=a['bbox']; g[a['image_id']].append((a['category_id'],[x,y,x+w,y+h]))
o=np.argsort(PI,kind='stable'); u,f0=np.unique(PI[o],return_index=True); sl=dict(zip(u.tolist(),np.split(o,f0[1:])))
out=[]
for im,gl in g.items():
    p=sl.get(im); 
    if p is None: continue
    for c,b in gl:
        q=p[PC[p]==c]
        if not len(q): continue
        B=PB[q]; b=np.array(b)
        iw=np.clip(np.minimum(B[:,2],b[2])-np.maximum(B[:,0],b[0]),0,None); ih=np.clip(np.minimum(B[:,3],b[3])-np.maximum(B[:,1],b[1]),0,None)
        I=iw*ih; U=(B[:,2]-B[:,0])*(B[:,3]-B[:,1])+(b[2]-b[0])*(b[3]-b[1])-I; iou=I/np.maximum(U,1e-9)
        j=np.argmax(PS[q]*(iou>=.5))
        if iou[j]>=.5: out.append((names[c],max(b[2]-b[0],b[3]-b[1]),iou[j],PS[q][j],np.abs(B[j]-b).mean()))
cn=np.array([r[0] for r in out]); L,IO,SC,E=[np.array([r[i] for r in out]) for i in (1,2,3,4)]
print('\nE000 val best-scoring TP per GT: median score by GT long side, within edge-error bins (px)')
for lo,hi in [(0,1),(1,2),(2,4)]:
    m=(E>=lo)&(E<hi); s=[]
    for a,b in [(0,8),(8,16),(16,32),(32,64),(64,1e9)]:
        k=m&(L>=a)&(L<b); s.append(f'{a}-{int(min(b,999))}:{np.median(SC[k]):.2f}/{np.median(IO[k]):.2f}(n{k.sum()})' if k.sum()>20 else f'{a}-:-')
    print(f'err {lo}-{hi}px  score/IoU', '  '.join(s))
