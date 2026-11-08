import json, struct, sys, re, urllib.request, collections
def header(repo, fn):
    u = f"https://huggingface.co/{repo}/resolve/main/{fn}"
    r = urllib.request.Request(u, headers={"Range": "bytes=0-7"}); n = struct.unpack("<Q", urllib.request.urlopen(r).read())[0]
    r = urllib.request.Request(u, headers={"Range": f"bytes=8-{7+n}"}); return json.loads(urllib.request.urlopen(r).read())
for repo in sys.argv[1:]:
    idx = json.load(open(repo.replace("/", "_") + ".index.json"))
    files = sorted(set(idx["weight_map"].values()))
    cat = collections.Counter(); dt = collections.Counter()
    for f in files:
        for k, v in header(repo, f).items():
            if k == "__metadata__": continue
            b = v["data_offsets"][1] - v["data_offsets"][0]
            dt[v["dtype"]] += b
            if "visual" in k or k.startswith("mtp") or ".mtp." in k: c = "skip(visual/mtp)"
            elif "embed_tokens" in k: c = "embed"
            elif "lm_head" in k: c = "lm_head"
            elif re.search(r"mlp\.experts\.", k): c = "routed_experts"
            elif "shared_expert" in k: c = "shared_expert"
            elif "linear_attn" in k: c = "gdn"
            elif "self_attn" in k: c = "attn"
            else: c = "other(norm/router)"
            cat[c] += b
    print(repo); [print(f"  {k:20s} {v/1e9:7.3f} GB") for k, v in sorted(cat.items())]
    print("  dtypes", {k: round(v/1e9, 2) for k, v in dt.items()})
    per_tok = cat["routed_experts"] * 8 / 256 + sum(v for k, v in cat.items() if k not in ("routed_experts", "embed", "skip(visual/mtp)"))
    print(f"  bytes/token {per_tok/1e9:.3f} GB -> floor {per_tok/240e9*1e3:.1f} ms at 240 GB/s; routed share {cat['routed_experts']*8/256/per_tok:.0%}")
