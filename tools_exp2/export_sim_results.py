import os, json, csv

workspace_root = r"c:\Users\ASUS\Desktop\Markov"
exp2_root = os.path.join(workspace_root, "exp2(generation_blind)")
root = os.path.join(exp2_root, "generation")
out_csv = os.path.join(exp2_root, "analysis_results", "simulation_summary.csv")

rows = []
for dirpath, _, files in os.walk(root):
    for f in files:
        if not f.endswith('.json'):
            continue
        p = os.path.join(dirpath, f)
        rel = os.path.relpath(p, workspace_root)
        try:
            with open(p, 'r', encoding='utf-8') as fh:
                j = json.load(fh)
        except Exception as e:
            rows.append({
                'path': rel,
                'success': 'read_error',
                'error': str(e),
            })
            continue
        row = {
            'path': rel,
            'success': bool(j.get('success')),
            'player1_id': j.get('player1_id',''),
            'player2_id': j.get('player2_id',''),
            'player1_name': j.get('player1_name',''),
            'player2_name': j.get('player2_name',''),
            'context_rounds': j.get('context_rounds',''),
            'simulate_rounds': j.get('simulate_rounds',''),
            'model': j.get('model',''),
            'api_type': j.get('api_type',''),
            'combo_type': j.get('combo_type',''),
            'parsed_rounds': j.get('llm_simulation',{}).get('parsed_rounds',''),
            'error': j.get('error',''),
        }
        rows.append(row)

os.makedirs(os.path.dirname(out_csv), exist_ok=True)
with open(out_csv, 'w', newline='', encoding='utf-8') as csvf:
    fieldnames = ['path','success','player1_id','player2_id','player1_name','player2_name',
                  'context_rounds','simulate_rounds','model','api_type','combo_type','parsed_rounds','error']
    writer = csv.DictWriter(csvf, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print('WROTE:', out_csv)
