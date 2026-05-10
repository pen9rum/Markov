import json
import glob
import os
import csv
from collections import Counter

ALL=('Rock','Paper','Scissors')
BEAT={'Rock':'Paper','Paper':'Scissors','Scissors':'Rock'}
LOSE={'Rock':'Scissors','Paper':'Rock','Scissors':'Paper'}


def exp_qrs(s,self_prev,opp_prev):
    if self_prev not in ALL or opp_prev not in ALL: return None
    if self_prev==opp_prev: return ALL
    m=[a for a in ALL if a!=self_prev and a!=opp_prev][0]
    return m if s=='Q' else (BEAT[m] if s=='R' else LOSE[m])


def exp_tuv(s,a,b,c,d):
    if any(x not in ALL for x in (a,b,c,d)): return None
    miss=[x for x in ALL if x not in {a,b,c,d}]
    if len(miss)==0: return ALL
    if len(miss)==1:
        m=miss[0]
        return m if s=='T' else (BEAT[m] if s=='U' else LOSE[m])
    if len(miss)==2:
        return tuple(miss) if s=='T' else (tuple(BEAT[m] for m in miss) if s=='U' else tuple(LOSE[m] for m in miss))
    return None


def analyze_file(path,setname,window_size=100,max_period=50,period_coverage_thresh=0.8):
    try:
        d=json.load(open(path,encoding='utf-8'))
    except Exception as e:
        return None
    p1,p2=d.get('player1_id'),d.get('player2_id')
    pred1=d.get('llm_simulation',{}).get('p1_identity')
    pred2=d.get('llm_simulation',{}).get('p2_identity')
    m1=d.get('llm_simulation',{}).get('p1_trajectory','').split()
    m2=d.get('llm_simulation',{}).get('p2_trajectory','').split()

    mark=set('QRS') if setname=='qrs' else set('TUV')
    if p1 in mark:
        mid, A, B, side = p1, m1, m2, 'p1'
    elif p2 in mark:
        mid, A, B, side = p2, m2, m1, 'p2'
    else:
        return None

    n=min(len(A),len(B))
    evals=0; wrong=0
    strict=[]
    # per-turn evals
    for i in range(n):
        if setname=='qrs':
            if i<1: continue
            e=exp_qrs(mid,A[i-1],B[i-1])
        else:
            if i<2: continue
            e=exp_tuv(mid,A[i-2],A[i-1],B[i-2],B[i-1])
        if e is None: continue
        evals += 1
        ok = (A[i] in e) if isinstance(e,tuple) else (A[i]==e)
        if not ok: wrong += 1

    # strict per window
    windows=[]
    for st in range(0,n,window_size):
        we=ww=0
        wseq=[]
        for i in range(st,min(st+window_size,n)):
            if setname=='qrs':
                if i<1: continue
                e=exp_qrs(mid,A[i-1],B[i-1])
            else:
                if i<2: continue
                e=exp_tuv(mid,A[i-2],A[i-1],B[i-2],B[i-1])
            if e is None: continue
            we += 1
            ok = (A[i] in e) if isinstance(e,tuple) else (A[i]==e)
            if not ok: ww += 1
            wseq.append(A[i])
        if we>0:
            strict.append(1 if ww==0 else 0)
            windows.append(tuple(wseq))

    if strict and all(x==1 for x in strict): tag='all1'
    elif strict and all(x==0 for x in strict): tag='all0'
    elif strict: tag='mixed'
    else: tag='none'

    # transitions: indices where strict changes (1->0 or 0->1). 1-based window numbers
    transitions=[i+1 for i in range(1,len(strict)) if strict[i]!=strict[i-1]]

    # longest run of identical moves in A
    longest_run=0
    cur=0; prev=None
    for x in A[:n]:
        if x==prev:
            cur += 1
        else:
            cur = 1
            prev = x
        if cur>longest_run: longest_run=cur

    # periodicity detection
    period_found=None; period_coverage=0.0
    for p in range(1, min(max_period, max(1,n))+1):
        # build repeated pattern
        pattern=A[:p]
        if not pattern: continue
        rep=(pattern * ((n//p)+2))[:n]
        match=sum(1 for i in range(n) if A[i]==rep[i]) if n>0 else 0
        cov=match/n if n>0 else 0.0
        if cov>=period_coverage_thresh:
            period_found=p; period_coverage=cov
            break

    # window identity fraction: fraction of windows identical to previous
    identical_prev=0
    for i in range(1,len(windows)):
        if windows[i]==windows[i-1]: identical_prev += 1
    window_identity_frac = identical_prev / max(1, len(windows)-1) if len(windows)>1 else 0.0

    # most common window frequency
    most_common_window_count = 0
    if windows:
        c=Counter(windows)
        most_common_window_count = c.most_common(1)[0][1]

    return {
        'file':path,
        'basename':os.path.basename(path),
        'markov':mid,
        'side':side,
        'evals':evals,
        'wrong':wrong,
        'overlap':(1 - (wrong/evals) if evals>0 else None),
        'strict_tag':tag,
        'strict_windows': ''.join(str(s) for s in strict),
        'transitions': transitions,
        'longest_run':longest_run,
        'period': period_found,
        'period_coverage': round(period_coverage,4),
        'window_identity_frac': round(window_identity_frac,4),
        'most_common_window_count': most_common_window_count
    }


def main():
    sets=['qrs','tuv']
    types=['type1_non_markov','type2_markov_p1','type3_markov_p2']
    out_dir='exp3(complex_markov)/analysis_results'
    os.makedirs(out_dir, exist_ok=True)

    for setname in sets:
        rows=[]
        for t in types:
            files=sorted(glob.glob(f'exp3(complex_markov)/generation/{setname}/deepseek-reasoner/ctx1000_sim1500/{t}/sim_*.json'))
            for f in files:
                r=analyze_file(f,setname)
                if r: r['type']=t; r['set']=setname; rows.append(r)

        csvpath=os.path.join(out_dir,f'repeat_strict_summary_{setname}.csv')
        keys=['file','basename','set','type','markov','side','evals','wrong','overlap','strict_tag','strict_windows','transitions','longest_run','period','period_coverage','window_identity_frac','most_common_window_count']
        with open(csvpath,'w',newline='',encoding='utf-8') as cf:
            w=csv.DictWriter(cf, fieldnames=keys)
            w.writeheader()
            for r in rows:
                rr={k:r.get(k) for k in keys}
                w.writerow(rr)

        print(f"\n[{setname}] files analyzed: {len(rows)}  -> summary: {csvpath}")
        # print a short summary sample
        for r in rows[:6]:
            print(f"  {r['basename']}: evals={r['evals']} wrong={r['wrong']} overlap={r['overlap']:.4f} strict={r['strict_tag']} transitions={r['transitions']} period={r['period']} win_id_frac={r['window_identity_frac']}")

if __name__=='__main__':
    main()
