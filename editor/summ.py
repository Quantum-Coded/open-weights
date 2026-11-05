import json,sys
for p in sys.argv[1:]:
    rs=[json.loads(l) for l in open(p)]
    print(p, 'acc',sum(r['correct'] for r in rs),'/',len(rs),'loop',sum(r['loop'] for r in rs),'cap',sum(r['status']=='cap' for r in rs))
    print(' keys',list(rs[0].keys()))
    for r in rs:
        print(f"  {r['id']:28s} c={int(r['correct'])} loop={int(r['loop'])} st={r['status']:8s} len={len(r['response'])}")
