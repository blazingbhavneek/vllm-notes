# Fetch the first `nbytes` of one tensor in a remote safetensors shard (range request).
import json, struct, sys, urllib.request
def rng(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
    with urllib.request.urlopen(req) as r:
        return r.read()
repo, path, name, nbytes, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
assert nbytes <= 8 * 2**20
url = f"https://huggingface.co/{repo}/resolve/main/{path}"
n = struct.unpack("<Q", rng(url, 0, 7))[0]
t = json.loads(rng(url, 8, 8 + n - 1))[name]
a = t["data_offsets"][0]
open(out, "wb").write(rng(url, 8 + n + a, 8 + n + a + nbytes - 1))
print(name, t["dtype"], t["shape"], "fetched", nbytes, "bytes")
