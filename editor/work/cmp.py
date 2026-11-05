import json,sys,numpy as np
a={r['id']:r for r in map(json.loads,open(sys.argv[1]))}
b={r['id']:r for r in map(json.loads,open(sys.argv[2]))}
for k in a:
    if a[k]['correct']!=b[k]['correct'] or a[k]['status']!=b[k]['status']:
        print(k[:30].ljust(30), 'A',int(a[k]['correct']),a[k]['status'],a[k]['n_tokens'],'-> B',int(b[k]['correct']),b[k]['status'],b[k]['n_tokens'])
print('acc',sum(r['correct'] for r in a.values()),'->',sum(r['correct'] for r in b.values()),' mean tok',np.mean([r['n_tokens'] for r in a.values()]).round(),'->',np.mean([r['n_tokens'] for r in b.values()]).round())
