"""Val-only: compare SARDet-100K boxes, RSAR boxes and E000 predictions by source. CPU only; reads RSAR.tar in place. Usage: rsar_offset_check.py RUN_DIR"""
import glob, json, os, sys, tarfile, multiprocessing as mp
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
BASE=Path('/root/zxy/TCRA-SAR/datasets'); ROOT=BASE/'SARDet_100K'
RUN=Path(sys.argv[1])
val=json.load(open(ROOT/'Annotations/val.json'))
names={c['id']:c['name'] for c in val['categories']}; nid={v:k for k,v in names.items()}
images={i['id']:i for i in val['images']}
def thumb(p):
    im=Image.open(p).convert('L'); t=np.asarray(im.resize((32,32),Image.BILINEAR),np.float32).ravel(); t-=t.mean(); n=np.linalg.norm(t)
    return (t/n if n>0 else t), im.size
def same(task):
    a,b=task; A=np.asarray(Image.open(a).convert('L'),np.float32); B=np.asarray(Image.open(b).convert('L'),np.float32)
    return A.shape==B.shape and np.abs(A-B).mean()<2
def iou(a,b):
    x1,y1=np.maximum(a[:,None,0],b[:,0]),np.maximum(a[:,None,1],b[:,1]); x2,y2=np.minimum(a[:,None,2],b[:,2]),np.minimum(a[:,None,3],b[:,3])
    inter=np.clip(x2-x1,0,None)*np.clip(y2-y1,0,None)
    return inter/np.maximum((a[:,2]-a[:,0])[:,None]*(a[:,3]-a[:,1])[:,None]+(b[:,2]-b[:,0])*(b[:,3]-b[:,1])-inter,1e-9)
def match(a,b,th=.3):
    if not len(a) or not len(b): return []
    m=iou(a,b); out=[]
    while m.size and m.max()>=th:
        i,j=np.unravel_index(m.argmax(),m.shape); out.append((i,j,m[i,j])); m[i,:]=-1; m[:,j]=-1
    return out
def main():
    paths={'ogsod':sorted(glob.glob(str(BASE/'OGSOD-1.0/sar/images/*/*'))),'msar':sorted(glob.glob(str(BASE/'MSAR-1.0 dataset/JPEGImages/*')))}
    vp=[str(ROOT/'JPEGImages/val'/i['file_name']) for i in val['images']]
    src=np.array(['other']*len(vp),dtype=object); ogsod_ref={}
    with mp.Pool(12) as pool:
        V=pool.map(thumb,vp,chunksize=64); Q=np.stack([r[0] for r in V])
        for ref,ps in paths.items():
            R=pool.map(thumb,ps,chunksize=64); Rm=np.stack([r[0] for r in R]); bs,bi=[],[]
            for s in range(0,len(Q),512):
                M=Q[s:s+512]@Rm.T; bs.append(M.max(1)); bi.append(M.argmax(1))
            bs,bi=np.concatenate(bs),np.concatenate(bi)
            cand=[(q,int(bi[q])) for q in np.flatnonzero(bs>=.98) if V[q][1]==R[bi[q]][1]]
            ok=pool.map(same,[(vp[q],ps[r]) for q,r in cand],chunksize=64)
            for (q,r),g in zip(cand,ok):
                if g:
                    src[q]=ref
                    if ref=='ogsod': ogsod_ref[q]=ps[r]
    print({k:int((src==k).sum()) for k in ('ogsod','msar','other')},flush=True)
    idx={i['id']:n for n,i in enumerate(val['images'])}
    gt=defaultdict(lambda: defaultdict(list))
    for a in val['annotations']:
        x,y,w,h=a['bbox']; gt[a['image_id']][a['category_id']].append([x,y,x+w,y+h])
    want={os.path.splitext(i['file_name'])[0]:i['id'] for i in val['images']}
    rs=defaultdict(lambda: defaultdict(list)); poly_over=0; n_poly=0
    with tarfile.open(BASE/'RSAR.tar','r:') as f:
        for m in f:
            if not m.name.startswith('val/annfiles/') or not m.name.endswith('.txt'): continue
            stem=Path(m.name).stem
            if stem not in want: continue
            rs_id=want[stem]; rs[rs_id]
            for line in f.extractfile(m).read().decode().splitlines():
                p=line.split()
                if len(p)<9: continue
                c=np.array(p[:8],float); xs,ys=c[0::2],c[1::2]
                rs[rs_id][nid[p[8]]].append([xs.min(),ys.min(),xs.max(),ys.max()])
    print('rsar val images matched',len(rs),'of',len(val['images']),flush=True)
    s=np.load(RUN/'independent_verify_20261106/val_predictions.npz'); PI,PC,PS,PB=s['image_id'],s['label'].astype(int),s['score'],s['box']
    k=PS>=.3; PI,PC,PB=PI[k],PC[k],PB[k]
    pred=defaultdict(lambda: defaultdict(list))
    for i,c,b in zip(PI.tolist(),PC.tolist(),PB): pred[i][c].append(b)
    rows=defaultdict(lambda: defaultdict(list))
    for iid in rs:
        grp=src[idx[iid]]; W,H=images[iid]['width'],images[iid]['height']
        for c in names:
            G=np.array(gt[iid].get(c,[]),float).reshape(-1,4); R=np.array(rs[iid].get(c,[]),float).reshape(-1,4); P=np.array(pred[iid].get(c,[]),float).reshape(-1,4)
            key=(grp,names[c])
            for gi,ri,v in match(G,R):
                g,r=G[gi],R[ri]; rows[key]['rs_gt_dx'].append((r[0]+r[2]-g[0]-g[2])/2); rows[key]['rs_gt_dy'].append((r[1]+r[3]-g[1]-g[3])/2); rows[key]['rs_gt_iou'].append(v)
                rows[key]['same'].append(float(np.abs(r-g).max()<=.5))
                if len(P):
                    pm=iou(P,np.stack([g]))[:,0]; j=int(pm.argmax())
                    if pm[j]>=.3:
                        p=P[j]; rows[key]['p_gt_dx'].append((p[0]+p[2]-g[0]-g[2])/2); rows[key]['p_rs_dx'].append((p[0]+p[2]-r[0]-r[2])/2); rows[key]['p_gt_dy'].append((p[1]+p[3]-g[1]-g[3])/2); rows[key]['p_rs_dy'].append((p[1]+p[3]-r[1]-r[3])/2)
                        rows[key]['p_gt_iou'].append(iou(p[None],g[None])[0,0]); rows[key]['p_rs_iou'].append(iou(p[None],r[None])[0,0])
                if c==nid['aircraft'] and (g[0]<-1 or g[1]<-1 or g[2]>W+1 or g[3]>H+1):
                    rows[key]['air_over_rs'].append(float(r[0]<-1 or r[1]<-1 or r[2]>W+1 or r[3]>H+1))
    def se(a): return float(np.std(a)/max(len(a)**.5,1))
    out={}
    for key in sorted(rows):
        r=rows[key]; o={'n':len(r['rs_gt_dx'])}
        for f in ('rs_gt_dx','rs_gt_dy','p_gt_dx','p_rs_dx','p_gt_dy','p_rs_dy'):
            if r[f]:
                a=np.array(r[f]); a=a[np.abs(a)<6]; o[f]=f'{a.mean():+.3f}+-{se(a):.3f}'
        for f in ('rs_gt_iou','p_gt_iou','p_rs_iou'):
            if r[f]: o[f+'_med']=round(float(np.median(r[f])),3)
        o['box_same_share']=round(float(np.mean(r['same'])),3)
        if r['air_over_rs']: o['air_overflow_also_in_rsar']=[len(r['air_over_rs']),round(float(np.mean(r['air_over_rs'])),3)]
        out['/'.join(key)]=o; print(key,o,flush=True)
    (RUN/'independent_verify_20261106/rsar').mkdir(exist_ok=True); json.dump(out,open(RUN/'independent_verify_20261106/rsar/report.json','w'),indent=1)
if __name__=='__main__': main()
