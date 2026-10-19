# Fetch ONE tensor's bytes from a remote safetensors shard with HTTP range requests.
# usage: st_tensor.py <repo> <revision> <shard path> <tensor name> <out.bin>
import json, struct, sys, urllib.request
def rng(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
    with urllib.request.urlopen(req) as r:
        return r.read()
repo, rev, path, name, out = sys.argv[1:6]
url = f"https://huggingface.co/{repo}/resolve/{rev}/{path}"
n = struct.unpack("<Q", rng(url, 0, 7))[0]
hdr = json.loads(rng(url, 8, 8 + n - 1))
t = hdr[name]
a, b = t["data_offsets"]
if b - a > 8 * 2**20:
    sys.exit(f"refusing: tensor is {(b-a)/2**20:.1f} MiB")
data = rng(url, 8 + n + a, 8 + n + b - 1) if b > a else b""
open(out, "wb").write(data)
json.dump({"dtype": t["dtype"], "shape": t["shape"]}, open(out + ".json", "w"))
print(name, t["dtype"], t["shape"], len(data), "bytes")
