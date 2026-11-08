import sys, gzip, json, collections
ev = json.load(gzip.open(sys.argv[1], "rt"))["traceEvents"]
t = collections.Counter(); n = collections.Counter()
steps = [e for e in ev if str(e.get("name","")).startswith("ProfilerStep#")]
for e in ev:
    if e.get("ph") == "X" and e.get("cat") == "kernel":
        t[e["name"][:90]] += e["dur"]; n[e["name"][:90]] += 1
tot = sum(t.values())
print(f"steps={len(steps)} total_kernel_ms={tot/1000:.1f} distinct={len(t)}")
for k, v in t.most_common(int(sys.argv[2])):
    print(f"{v/1000:9.2f} ms {n[k]:6d}x {v/tot:6.1%}  {k}")
