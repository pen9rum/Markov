import os, json
root = r"c:\Users\ASUS\Desktop\Markov\exp2(generation_blind)\generation"

total = 0
succ = 0
fail = 0
fails = []
for dirpath, _, files in os.walk(root):
    for f in files:
        if f.endswith('.json'):
            total += 1
            p = os.path.join(dirpath, f)
            try:
                with open(p, 'r', encoding='utf-8') as fh:
                    j = json.load(fh)
                if j.get('success'):
                    succ += 1
                else:
                    fail += 1
                    fails.append(p)
            except Exception as e:
                fail += 1
                fails.append(p + ' (read error: ' + str(e) + ')')

print('TOTAL_JSON:', total)
print('SUCCESS:', succ)
print('FAIL:', fail)
if fails:
    print('\nFAILED FILES:')
    for p in fails:
        print(p)
