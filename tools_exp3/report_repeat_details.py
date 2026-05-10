import csv
import ast
import os
from check_repeats_and_strict_windows import analyze_file

CSV_DIR='exp3(complex_markov)/analysis_results'
QCSV=os.path.join(CSV_DIR,'repeat_strict_summary_qrs.csv')
TCSV=os.path.join(CSV_DIR,'repeat_strict_summary_tuv.csv')


def read_csv(path):
    rows=[]
    if not os.path.exists(path):
        return rows
    with open(path,encoding='utf-8') as f:
        r=csv.DictReader(f)
        for row in r:
            # parse numeric and list-like fields
            for k in ['transitions','period','window_identity_frac','most_common_window_count','evals','wrong']:
                if k in row:
                    val=row[k]
                    try:
                        row[k]=ast.literal_eval(val)
                    except Exception:
                        try:
                            row[k]=float(val)
                        except Exception:
                            row[k]=val
            rows.append(row)
    return rows


def get_windows_content(path,setname,window_size=100,max_windows=6):
    # replicate windows extraction to show actual moves in each window
    import json
    d=json.load(open(path,encoding='utf-8'))
    p1,p2=d.get('player1_id'),d.get('player2_id')
    m1=d.get('llm_simulation',{}).get('p1_trajectory','').split()
    m2=d.get('llm_simulation',{}).get('p2_trajectory','').split()
    mark=set('QRS') if setname=='qrs' else set('TUV')
    if p1 in mark:
        A,B = m1,m2
    else:
        A,B = m2,m1
    n=min(len(A),len(B))
    windows=[]
    for st in range(0,n,window_size):
        wseq=[]
        for i in range(st,min(st+window_size,n)):
            wseq.append(A[i])
        if wseq:
            windows.append(wseq)
        if len(windows)>=max_windows:
            break
    return windows


def report(setname,csvpath):
    rows=read_csv(csvpath)
    print('\n'+'='*80)
    print(setname.upper(),'REPORT')
    print('='*80)
    if not rows:
        print('  (no summary CSV)')
        return
    trans_files=[r for r in rows if isinstance(r.get('transitions'), list) and len(r.get('transitions'))>0]
    period_files=[r for r in rows if r.get('period') not in (None,'None','')]
    high_identity=[r for r in rows if isinstance(r.get('window_identity_frac'), float) and r.get('window_identity_frac')>=0.7]

    print(f"  total files in CSV: {len(rows)}")
    print(f"  files with transitions (strict changes): {len(trans_files)}")
    for r in trans_files[:10]:
        print(f"    {r['basename']}  transitions={r['transitions']} strict_windows={r['strict_windows']}")
    print(f"  files with detected period: {len(period_files)}")
    for r in period_files[:10]:
        print(f"    {r['basename']}  period={r['period']} period_coverage={r.get('period_coverage')}")
    print(f"  files with high window identity (>=0.7): {len(high_identity)}")
    for r in high_identity[:10]:
        print(f"    {r['basename']}  win_id_frac={r['window_identity_frac']} strict={r['strict_windows']}")

    # show detailed window content for up to 2 example files from each category
    examples=[]
    if trans_files:
        examples.extend([(r['basename'],r['file']) for r in trans_files[:2]])
    if period_files:
        examples.extend([(r['basename'],r['file']) for r in period_files[:2]])
    if high_identity:
        examples.extend([(r['basename'],r['file']) for r in high_identity[:2]])

    if not examples:
        print('  (no example files to show)')
        return

    print('\n  Detailed window samples:')
    for name,path in examples:
        try:
            windows=get_windows_content(path,setname)
            print(f"\n    {name} -> first {len(windows)} windows:")
            for wi,w in enumerate(windows,1):
                print(f"      W{wi:02d} (len={len(w)}): {' '.join(w[:40])}{' ...' if len(w)>40 else ''}")
        except Exception as e:
            print('      Error reading',name,e)

if __name__=='__main__':
    report('qrs',QCSV)
    report('tuv',TCSV)
