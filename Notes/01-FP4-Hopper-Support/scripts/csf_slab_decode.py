# Usage (see note 21, section "Reproduce the checks"):
#   1. Fetch one Qwen expert with st_tensor.py into a folder DIR (about 1 MB).
#   2. python csf_slab_decode.py DIR
# Needs numpy. Verified on 2026-10-19 against nvidia/Qwen3.8-Flash-Next-FP4-Local.

# Decode slab CSF ("csf") scales with numpy and compare against a raw scale tensor.
import json, sys
import numpy as np

def load(p):
    meta = json.load(open(p + ".json"))
    dt = {"U8": np.uint8, "U32": np.uint32, "F32": np.float32, "F8_E4M3": np.uint8}[meta["dtype"]]
    return np.fromfile(p, dtype=dt).reshape(meta["shape"])

def decode_slab(fixed, exc):
    rows = fixed.shape[0] * 16
    cols = (fixed.shape[1] // 16 - 1) * 2
    bases = fixed[:, :16].reshape(rows, 1).astype(np.int32)
    assert bases.max() <= 240
    packed = fixed[:, 16:].reshape(rows, cols // 2)
    codes = np.stack((packed & 0xF, packed >> 4), axis=-1).reshape(rows, cols)
    out = (bases + codes).reshape(-1)
    pos = exc & 0xFFFFFF
    assert np.all(np.diff(pos.astype(np.int64)) > 0), "exceptions must be sorted and unique"
    out[pos] = exc >> 24
    return out.astype(np.uint8).reshape(rows, cols)

d = sys.argv[1]
fixed, exc = load(f"{d}/qloc_down.fixed.bin"), load(f"{d}/qloc_down.exc.bin")
s_loc = decode_slab(fixed, exc)
s_ptq = load(f"{d}/qptq_down.scale.bin")
print("decoded FP4-Local scale shape", s_loc.shape, "PTQ scale shape", s_ptq.shape)
print("exceptions:", exc.size, f"= {exc.size / s_loc.size:.3%} of scales")
print("FP4-Local scales identical to the PTQ NVFP4 release?", np.array_equal(s_loc, s_ptq))
w_loc, w_ptq = load(f"{d}/qloc_down.weight.bin"), load(f"{d}/qptq_down.weight.bin")
print("FP4 nibbles identical?", np.array_equal(w_loc, w_ptq), f"(bytes equal: {np.mean(w_loc == w_ptq):.1%})")
print("global scales (weight_scale_2): FP4-Local", load(f"{d}/qloc_down.ws2.bin"), "PTQ", load(f"{d}/qptq_down.ws2.bin"))
raw = s_loc.size + 0
csf = fixed.nbytes + exc.nbytes
print(f"scale bytes raw {raw} vs CSF {csf} -> {csf/raw:.1%}")
np.save(f"{d}/qloc_down.scale_decoded.npy", s_loc)

# Re-encode with the base rule from csf32_vectorized.best_bases and compare bytes.
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from csf32_vectorized import best_bases
rows, cols = s_loc.shape
b = best_bases(s_loc.astype(np.int32))
delta = s_loc.astype(np.int32) - b[:, None]
outlier = (delta < 0) | (delta > 15)
delta[outlier] = 0
packed = (delta[:, 0::2] | (delta[:, 1::2] << 4)).astype(np.uint8)
fixed2 = np.concatenate((b.astype(np.uint8).reshape(rows // 16, 16), packed.reshape(rows // 16, -1)), 1)
pos = np.flatnonzero(outlier)
exc2 = ((s_loc.reshape(-1)[pos].astype(np.uint64) << 24) | pos.astype(np.uint64)).astype(np.uint32)
print("re-encoding reproduces NVIDIA's fixed and exception streams:",
      np.array_equal(fixed2, fixed), np.array_equal(exc2, exc))
