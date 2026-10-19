# Usage (see note 21, section "Reproduce the checks"):
#   1. Fetch the five csf32_* streams of one GLM expert plus the uncompressed
#      weight_scale from the LIL branch qad2-kda-w8a16 (about 800 KB) into DIR.
#   2. python csf32_decode.py DIR   -> prints "decoded == uncompressed QAD scale: True"
# Needs numpy. Verified on 2026-10-19 against nvidia/GLM-5.3-Flash-FP4-Local.

# Independent, simple decoder for csf_32x2_v1 NVFP4 scales; checked on real data.
import json, sys
import numpy as np

def load(p):
    meta = json.load(open(p + ".json"))
    dt = {"U8": np.uint8, "U16": np.uint16, "U32": np.uint32, "F8_E4M3": np.uint8}[meta["dtype"]]
    return np.fromfile(p, dtype=dt).reshape(meta["shape"])

def decode_csf32(bases, codes, page_bases, offsets, exceptions, rows, cols):
    """Return the [rows, cols] uint8 scale bytes (E4M3 bit patterns)."""
    assert rows % 32 == 0
    col_blocks = (cols + 1) // 2           # blocks are 2 columns wide
    row_blocks = rows // 32                # blocks are 32 rows tall
    # 1) The order blocks are stored in: 128x4 tiles (4 row-blocks x 2 col-blocks)
    #    row-major; inside a tile, the first column-block's row-blocks first.
    order = []                             # order[i] = (row_block, col_block)
    tile_sizes = []
    for rb0 in range(0, row_blocks, 4):
        for cb0 in range(0, col_blocks, 2):
            n = 0
            for cb in range(cb0, min(cb0 + 2, col_blocks)):
                for rb in range(rb0, min(rb0 + 4, row_blocks)):
                    order.append((rb, cb)); n += 1
            tile_sizes.append(n)
    out = np.zeros((rows, cols), dtype=np.int32)
    # 2) Codes: 4 bits per scale, row-major inside each block, low nibble first.
    nib = np.stack((codes & 15, codes >> 4), axis=-1).reshape(-1)
    pos = 0
    for rb, cb in order:
        width = min(2, cols - 2 * cb)      # last block may be 1 column wide
        for r in range(32):
            for c in range(width):
                out[rb * 32 + r, cb * 2 + c] = nib[pos]; pos += 1
    assert pos == nib.size
    # 3) Add the per-row base.
    assert bases.max() <= 240
    out += bases.astype(np.int32)[:, None]
    # 4) Exceptions: pages of 32 tiles; each page has (its blocks + 1) prefixes.
    block_i = 0; idx = 0
    tiles = len(tile_sizes)
    for page, t0 in enumerate(range(0, tiles, 32)):
        nblocks = sum(tile_sizes[t0:t0 + 32])
        prefix = offsets[idx: idx + nblocks + 1].astype(np.int64)
        assert prefix[0] == 0
        rec0 = int(page_bases[page]) // 2  # page base counts bytes; records are 2 bytes
        for j in range(nblocks):
            rb, cb = order[block_i + j]
            for k in range(prefix[j], prefix[j + 1]):
                rec = int(exceptions[rec0 + k])
                local, value = rec & 255, rec >> 8
                out[rb * 32 + local // 2, cb * 2 + local % 2] = value
        block_i += nblocks; idx += nblocks + 1
    assert block_i == len(order) and idx == offsets.size
    return out.astype(np.uint8)


if __name__ == "__main__":
    d = sys.argv[1]
    g = lambda s: load(f"{d}/glm_gate.{s}.bin")
    raw = g("raw_scale")
    dec = decode_csf32(g("csf32_bases"), g("csf32_codes"), g("csf32_page_bases"),
                       g("csf32_offsets"), g("csf32_exceptions"), *raw.shape)
    print("shape", raw.shape, "exceptions", g("csf32_exceptions").size,
          f"({g('csf32_exceptions').size / raw.size:.3%})")
    print("decoded == uncompressed QAD scale:", np.array_equal(dec, raw))
    csf_bytes = sum(g(s).nbytes for s in ["csf32_bases", "csf32_codes", "csf32_page_bases", "csf32_offsets", "csf32_exceptions"])
    print(f"CSF bytes {csf_bytes} vs raw {raw.nbytes} -> {csf_bytes / raw.nbytes:.1%}")
