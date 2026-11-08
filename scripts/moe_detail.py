import sys, gzip, json, collections, statistics as st
ev = json.load(gzip.open(sys.argv[1], "rt"))["traceEvents"]
d = collections.defaultdict(list); grid = {}
for e in ev:
    if e.get("ph")=="X" and e.get("cat")=="kernel" and (e["name"].startswith("_matmul_NNT") or e["name"]=="fused_moe_kernel"):
        d[e["name"]].append(e["dur"]); a = e.get("args", {})
        grid[e["name"]] = (a.get("grid"), a.get("block"))
for k, v in d.items():
    print(f"{k[:45]:45s} n={len(v)} median {st.median(v):7.1f} us  grid/block {grid[k]}")
