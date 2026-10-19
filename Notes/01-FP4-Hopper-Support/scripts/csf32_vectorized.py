# Usage (see note 21, section "Reproduce the checks"):
#   python csf32_vectorized.py            -> synthetic round-trip checks only
#   python csf32_vectorized.py DIR        -> also checks the real GLM tensor in DIR
#                                            (same files as csf32_decode.py)
# Needs numpy. Reference for the vLLM port (note 20, task T3): a vectorized
# csf_32x2_v1 decoder plus a test-only encoder, both checked against the
# simple loop decoder in csf32_decode.py.
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csf32_decode import decode_csf32  # noqa: E402  (runs nothing on import)

TILE_ROW_BLOCKS, TILE_COL_BLOCKS, TILES_PER_PAGE = 4, 2, 32


def block_order(rows, cols):
    """(row_block, col_block) per stored block, and the block count per tile."""
    rbs, cbs = rows // 32, (cols + 1) // 2
    order, tile_sizes = [], []
    for rb0 in range(0, rbs, TILE_ROW_BLOCKS):
        for cb0 in range(0, cbs, TILE_COL_BLOCKS):
            n = 0
            for cb in range(cb0, min(cb0 + TILE_COL_BLOCKS, cbs)):
                for rb in range(rb0, min(rb0 + TILE_ROW_BLOCKS, rbs)):
                    order.append((rb, cb)); n += 1
            tile_sizes.append(n)
    return np.array(order, dtype=np.int64).reshape(-1, 2), np.array(tile_sizes)


def code_positions(rows, cols, order):
    """Flat output index (row * cols + col) of every code, in stream order.

    Assumes cols is even (NVFP4: cols = K / 16 with K a multiple of 32), so every
    block is 2 columns wide. Compute once per shape and cache it.
    """
    assert rows % 32 == 0 and cols % 2 == 0
    r = np.arange(32).repeat(2)                      # 32 rows x 2 cols, row-major
    c = np.tile(np.arange(2), 32)
    rr = order[:, :1] * 32 + r                       # [blocks, 64]
    cc = order[:, 1:] * 2 + c
    return (rr * cols + cc).reshape(-1)


def page_spans(tile_sizes):
    """Block count of each page (32 tiles per page)."""
    return np.add.reduceat(tile_sizes, np.arange(0, len(tile_sizes), TILES_PER_PAGE))


def decode_vec(bases, codes, page_bases, offsets, exceptions, rows, cols):
    order, tile_sizes = block_order(rows, cols)
    pos = code_positions(rows, cols, order)
    out = np.empty(rows * cols, dtype=np.int32)
    out[pos] = np.stack((codes & 15, codes >> 4), -1).reshape(-1)
    out = out.reshape(rows, cols) + bases.astype(np.int32)[:, None]
    nblocks = page_spans(tile_sizes)
    idx0 = np.concatenate(([0], np.cumsum(nblocks + 1)[:-1]))
    blk0 = np.concatenate(([0], np.cumsum(nblocks)[:-1]))
    for p in range(len(nblocks)):
        prefix = offsets[idx0[p]: idx0[p] + nblocks[p] + 1].astype(np.int64)
        recs = exceptions[page_bases[p] // 2 + np.arange(prefix[-1])].astype(np.int64)
        blk = blk0[p] + np.repeat(np.arange(nblocks[p]), np.diff(prefix))
        local = recs & 255
        out[order[blk, 0] * 32 + local // 2, order[blk, 1] * 2 + local % 2] = recs >> 8
    return out.astype(np.uint8)


def best_bases(raw):
    """Per row, the base in 0..240 whose 16-value window covers the most scales."""
    hist = np.zeros((raw.shape[0], 256), np.int64)
    np.add.at(hist, (np.arange(raw.shape[0])[:, None], raw), 1)
    csum = np.concatenate((np.zeros((raw.shape[0], 1), np.int64), hist.cumsum(1)), 1)
    covered = csum[:, 16:257] - csum[:, 0:241]       # window [b, b + 15], b = 0..240
    return covered.argmax(1)                         # ties -> smallest base


def encode(scale):
    """Test-only encoder: per-row base chosen to minimize exceptions."""
    rows, cols = scale.shape
    raw = scale.astype(np.int32)
    bases = best_bases(raw)
    delta = raw - bases[:, None]
    outlier = (delta < 0) | (delta > 15)
    delta[outlier] = 0
    order, tile_sizes = block_order(rows, cols)
    pos = code_positions(rows, cols, order)
    nib = delta.reshape(-1)[pos].astype(np.uint8)
    codes = nib[0::2] | (nib[1::2] << 4)
    recs, offsets, page_bases = [], [], []
    b = 0
    for n in page_spans(tile_sizes):
        page_bases.append(2 * len(recs))
        count = 0
        offsets.append(0)
        for rb, cb in order[b: b + n]:
            blk = outlier[rb * 32: rb * 32 + 32, cb * 2: cb * 2 + 2]
            for lr, lc in zip(*np.nonzero(blk)):
                recs.append(int(raw[rb * 32 + lr, cb * 2 + lc]) << 8 | (2 * lr + lc))
                count += 1
            offsets.append(count)
        b += n
    return (bases.astype(np.uint8), codes, np.array(page_bases, np.uint32),
            np.array(offsets, np.uint16), np.array(recs, np.uint16))


def check(scale, label):
    streams = encode(scale)
    rows, cols = scale.shape
    vec = decode_vec(*streams, rows, cols)
    loop = decode_csf32(*streams, rows, cols)
    assert np.array_equal(vec, scale) and np.array_equal(loop, scale), label
    print(f"{label}: shape {scale.shape}, {streams[4].size} exceptions, "
          f"offsets {streams[3].size}, pages {streams[2].size} -> round trip OK")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for shape in [(2048, 256), (4096, 128), (96, 6), (32, 2), (160, 10)]:
        s = rng.integers(0x40, 0x50, shape).astype(np.uint8)
        s.reshape(-1)[rng.choice(s.size, max(1, s.size // 100), replace=False)] = 0x7E
        s.reshape(-1)[rng.choice(s.size, max(1, s.size // 200), replace=False)] = 0x08
        check(s, "synthetic")
    if len(sys.argv) > 1:
        d = sys.argv[1]

        def load(n):
            meta = json.load(open(f"{d}/glm_gate.{n}.bin.json"))
            dt = {"U8": np.uint8, "U16": np.uint16, "U32": np.uint32,
                  "F8_E4M3": np.uint8}[meta["dtype"]]
            return np.fromfile(f"{d}/glm_gate.{n}.bin", dt).reshape(meta["shape"])

        raw = load("raw_scale")
        real = [load(f"csf32_{n}") for n in
                ["bases", "codes", "page_bases", "offsets", "exceptions"]]
        print("real GLM tensor: vectorized decode == uncompressed scale:",
              np.array_equal(decode_vec(*real, *raw.shape), raw))
        mine = encode(raw)
        same = [np.array_equal(a, b) for a, b in zip(mine, real)]
        print("test encoder reproduces NVIDIA's streams (bases, codes, page_bases,"
              " offsets, exceptions):", same)
        check(raw, "real GLM scale re-encoded")
