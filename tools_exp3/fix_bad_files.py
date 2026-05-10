"""
Repeatedly delete bad files and re-run until all pass or max_attempts reached.
A file is "bad" if: no P1_identity OR parsed_rounds < 1000.
"""
import json, glob, re, os, subprocess, sys

EXP3_ROOT = os.path.join(os.path.dirname(__file__), '..', 'exp3(complex_markov)', 'generation')

TYPES = [
    ('type2', 'type2_markov_p1'),
    ('type3', 'type3_markov_p2'),
]

MAX_ATTEMPTS = 40
MIN_PARSED   = 1000
CONTEXT      = 1000
SIMULATE     = 1500
MODEL        = 'deepseek-reasoner'


def find_bad_files(markov_set: str) -> list:
    """Return list of (filepath, p1, p2, typ) for bad files."""
    bad = []
    for typ, typdir in TYPES:
        d = os.path.join(EXP3_ROOT, markov_set, MODEL, f'ctx{CONTEXT}_sim{SIMULATE}', typdir)
        for f in glob.glob(os.path.join(d, 'sim_*.json')):
            try:
                data = json.load(open(f, encoding='utf-8'))
                llm  = data.get('llm_simulation', {})
                parsed = llm.get('parsed_rounds', 0)
                raw    = llm.get('raw_output', '')
                has_id = bool(re.search(r'P1_identity\s*:\s*([A-Za-z])', raw))
                if not has_id or parsed < MIN_PARSED:
                    m = re.search(r'sim_([A-Za-z])_vs_([A-Za-z])_', f)
                    if m:
                        bad.append((f, m.group(1), m.group(2), typ))
            except Exception as e:
                print(f'  [WARN] Could not read {f}: {e}')
    return bad


def run_single(markov_set: str, p1: str, p2: str) -> bool:
    cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), 'run_generation.py'),
        '--markov-set', markov_set,
        '--p1', p1, '--p2', p2,
        '--context',  str(CONTEXT),
        '--simulate', str(SIMULATE),
        '--model',    MODEL,
    ]
    print(f'    Running: {" ".join(cmd[2:])}')
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    markov_sets = ['qrs', 'tuv', 'xyz_opp']

    for markov_set in markov_sets:
        print(f'\n{"="*60}')
        print(f'  Markov set: {markov_set.upper()}')
        print(f'{"="*60}')

        for attempt in range(1, MAX_ATTEMPTS + 1):
            bad = find_bad_files(markov_set)
            if not bad:
                print(f'  [OK] All files good after {attempt - 1} attempt(s).')
                break

            print(f'\n  Attempt {attempt}/{MAX_ATTEMPTS}: {len(bad)} bad file(s)')
            for fpath, p1, p2, typ in bad:
                reason = ''
                try:
                    data = json.load(open(fpath, encoding='utf-8'))
                    llm  = data.get('llm_simulation', {})
                    parsed = llm.get('parsed_rounds', 0)
                    raw    = llm.get('raw_output', '')
                    has_id = bool(re.search(r'P1_identity\s*:\s*([A-Za-z])', raw))
                    reason = 'no_id' if not has_id else f'parsed={parsed}'
                except:
                    reason = 'unreadable'
                print(f'    DELETE {p1}_vs_{p2} ({typ}) [{reason}]: {os.path.basename(fpath)}')
                os.remove(fpath)

            for _, p1, p2, _ in bad:
                run_single(markov_set, p1, p2)

        else:
            remaining = find_bad_files(markov_set)
            if remaining:
                print(f'\n  [FAIL] Still {len(remaining)} bad after {MAX_ATTEMPTS} attempts:')
                for _, p1, p2, typ in remaining:
                    print(f'    {p1}_vs_{p2} ({typ})')

    print('\nDone.')


if __name__ == '__main__':
    main()
