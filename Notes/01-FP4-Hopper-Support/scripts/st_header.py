# Fetch only the JSON header of a remote safetensors file via HTTP range requests.
import json, struct, sys, urllib.request
def rng(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
    with urllib.request.urlopen(req) as r:
        return r.read()
repo, path, out = sys.argv[1], sys.argv[2], sys.argv[3]
url = f"https://huggingface.co/{repo}/resolve/main/{path}"
n = struct.unpack("<Q", rng(url, 0, 7))[0]
hdr = json.loads(rng(url, 8, 8 + n - 1))
json.dump(hdr, open(out, "w"))
print(path, "header bytes", n, "tensors", len(hdr) - ("__metadata__" in hdr))
