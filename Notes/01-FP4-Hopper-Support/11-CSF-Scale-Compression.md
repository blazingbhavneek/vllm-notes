# 11 · CSF scale compression

**What you'll learn:** why NVFP4 block scales compress well; exactly how the two CSF formats store them, byte by byte, with worked examples; how big the savings are; and how to decode both formats (the decoders in `scripts/` are checked on real data). If you implement Phase A or Phase C, this note is your spec.

CSF is lossless. Decoding gives back **exactly** the original E4M3 scale bytes, so any NVFP4 kernel can use the result unchanged.

---

## 1. Why scales compress

An NVFP4 matrix `[R, K]` has a scale matrix `[R, C]` with C = K / 16: one E4M3 byte per 16 weights. That's 0.5 bits per weight, about 11 % of the file.

Two facts make these bytes compressible:

1. **For positive E4M3 numbers, a bigger byte means a bigger value** (note 02). Adding 8 to the byte doubles the value. So a window of 16 consecutive byte values covers a value range of 2 doublings (×4).
2. **Scales in one row are similar.** ✅ Measured on real FP4-Local experts: inside one row, the scales span a median of 1.54 doublings (max 3.36) for a Qwen `down_proj`, and a median of 2.0 (max 3.0) for a GLM `gate_proj`.

So for each row: pick a **base** byte. Most scales in the row are between `base` and `base + 15`, so store each as a **4-bit code** `byte − base`. That halves the size. The few scales outside the window are stored separately as **exceptions**.

### Worked example (one row, 6 scales)

```
scale bytes:   0x52  0x55  0x5A  0x61  0x70  0x53
choose base:   0x52
codes:         0     3     8     15    (18: too big!)  1
               → the 0x70 entry becomes an exception: (position 4, value 0x70)
               → its code slot holds any value (the encoder writes 0)
decode:        base + code for every slot, then overwrite slot 4 with 0x70
```

### How the encoder picks the base (✅ reproduced byte for byte)

For each row: try every base `b` from 0 to 240, count how many of the row's scales fall in `[b, b + 15]`, and pick the `b` with the highest count (on ties, the **smallest** `b`). Code slots of exceptions hold 0.

Re-encoding the real tensors with this rule gives **exactly** NVIDIA's bytes, for both formats: the Qwen slab streams (`scripts/csf_slab_decode.py`) and all five GLM `csf32` streams (`scripts/csf32_vectorized.py`). So tests can generate realistic CSF data from any scale matrix without shipping checkpoint bytes.

## 2. Variant 1: slab "csf" (Qwen FP4-Local, LIL GLM `main`)

`weight_scale_encoding: "csf"`, codec name `byte-window4-fixed-stream-u24-exceptions/1`. Each scale matrix `<name>.weight_scale` becomes **two** tensors:

### `<name>.weight_scale.nvfp4_csf_fixed`: uint8 `[R/16, 16 × (1 + C/2)]`

Rows are grouped in **slabs of 16**. Each slab row of the tensor holds:

```
[ base_0 base_1 … base_15 | row 0's codes (C/2 bytes) | row 1's codes | … | row 15's codes ]
   16 bytes                 two codes per byte, LOW nibble = even column
```

- `base_r` is one byte per scale row, at most **240** (so base + 15 ≤ 255).
- Byte `j` of a row's codes holds column `2j` in its low nibble and column `2j + 1` in its high nibble.

### `<name>.weight_scale.nvfp4_csf_exceptions`: uint32 `[E]`

Each 32-bit word:

```
bits 0–23  : flat position  = row × C + col
bits 24–31 : the true scale byte for that position
```

Words are sorted by position and each position appears once.

### Decoding (numpy, from `scripts/csf_slab_decode.py`)

```python
rows = fixed.shape[0] * 16
cols = (fixed.shape[1] // 16 - 1) * 2
bases = fixed[:, :16].reshape(rows, 1).astype(np.int32)          # step 1
packed = fixed[:, 16:].reshape(rows, cols // 2)                   # step 2
codes = np.stack((packed & 0xF, packed >> 4), axis=-1).reshape(rows, cols)
out = (bases + codes).reshape(-1)                                 # step 3
out[exc & 0xFFFFFF] = exc >> 24                                   # step 4
scale = out.astype(np.uint8).reshape(rows, cols)                  # E4M3 bytes
```

Shapes check (✅ from headers): Qwen `gate_proj` scale `[640, 160]` → fixed `[40, 1296]`; `down_proj` scale `[2560, 40]` → fixed `[160, 336]`.

✅ PR [#60646](https://github.com/vllm-project/vllm/pull/60646) (draft) implements exactly this in `vllm/model_executor/model_loader/csf_scales.py`, with a reference encoder in its test (row base = row minimum, capped at 240). That encoder picks bases differently from NVIDIA's (section 1), which is fine: any base gives valid, lossless streams, and the decoder doesn't care.

## 3. Variant 2: `csf_32x2_v1` (NVIDIA GLM FP4-Local, LIL `qad2-kda-w8a16-csf`)

This one is designed so a **GPU** can decode any part of it independently and in parallel. Each scale matrix becomes **five** tensors:

| Tensor | dtype | Length | Meaning |
|---|---|---|---|
| `.weight_scale.csf32_bases` | uint8 | R | one base per scale row (≤ 240) |
| `.weight_scale.csf32_codes` | uint8 | R × C / 2 | 4-bit codes, in **block order** (below) |
| `.weight_scale.csf32_page_bases` | uint32 | P | for each page: the **byte** offset of its first exception record |
| `.weight_scale.csf32_offsets` | uint16 | B + P | per page: a prefix count for each block, plus one closing entry |
| `.weight_scale.csf32_exceptions` | uint16 | E | low byte = position inside the block (2 × local_row + local_col, 0–63); high byte = the true scale byte |

### Blocks, tiles and pages

1. **Block** = 32 rows × 2 columns of the scale matrix (64 scales, 32 code bytes). There are B = (R / 32) × ⌈C / 2⌉ blocks.
2. **Tile** = 4 × 2 blocks = 128 rows × 4 columns.
3. Tiles are visited **row-major** (left to right, then down). Inside a tile, blocks are listed **first column-pair first**: the 4 row-blocks of the left column pair, then the 4 of the right one.
4. Inside a block, codes are row-major (row 0 col 0, row 0 col 1, row 1 col 0, …), two per byte, low nibble first.
5. **Page** = 32 consecutive tiles. Offsets are 16-bit, so counting restarts per page; `page_bases` gives each page's absolute start.

### Decoding (from `scripts/csf32_decode.py`, simplified)

```python
order = block_order(rows, cols)                 # list of (row_block, col_block), as above
nib = np.stack((codes & 15, codes >> 4), -1).reshape(-1)
# 1) codes into place
pos = 0
for rb, cb in order:
    for r in range(32):
        for c in range(min(2, cols - 2 * cb)):
            out[rb*32 + r, cb*2 + c] = nib[pos]; pos += 1
# 2) add the row base
out += bases[:, None]
# 3) exceptions, page by page
block_i = idx = 0
for page in range(num_pages):
    nblocks = blocks in this page's 32 tiles
    prefix = offsets[idx : idx + nblocks + 1]   # prefix[0] == 0
    rec0 = page_bases[page] // 2                # bytes → 2-byte records
    for j in range(nblocks):
        rb, cb = order[block_i + j]
        for k in range(prefix[j], prefix[j + 1]):
            rec = exceptions[rec0 + k]
            local, value = rec & 255, rec >> 8
            out[rb*32 + local // 2, cb*2 + local % 2] = value
    block_i += nblocks; idx += nblocks + 1
```

Shapes check (✅ GLM `gate_proj`, scale `[2048, 256]`): blocks B = 64 × 128 = 8192; tiles = 16 × 64 = 1024; pages P = 32; offsets = 8192 + 32 = **8224** ✅; codes = 2048 × 256 / 2 = **262 144** ✅.

✅ **Verified:** decoding NVIDIA GLM FP4-Local `layers.3.mlp.experts.0.gate_proj` gives exactly the bytes of the uncompressed scale in LIL's `qad2-kda-w8a16` branch (`decoded == uncompressed QAD scale: True`).

A **vectorized** version (no Python loop over blocks, only a short loop over pages) is in `scripts/csf32_vectorized.py`, together with a test-only encoder. It's checked against the loop decoder on synthetic shapes, including partial tiles (96 × 6, 160 × 10), and on the real tensor. Port this one to torch for vLLM.

No vLLM code (upstream or PR) reads this variant yet.

### Why a GPU likes this layout

- Every block's codes sit at a fixed, computable offset: `block_index × 32` bytes. No searching.
- Every block's exceptions are found with two lookups (`page_bases[page]`, `offsets[...]`).
- So one thread block can decode one tile, or one warp one block, with no dependence on the rest of the matrix. That's what makes "decode only the experts this batch uses" (Phase C) practical.

## 4. Sizes

Formulas (R rows, C scale columns, E exceptions):

```
raw scale          = R × C bytes
slab csf           = R × (1 + C/2) + 4E           → ≈ 50 % + 1/C + exceptions
csf_32x2_v1        = R + R×C/2 + 4P + 2(B + P) + 2E
```

✅ Real tensors:

| Tensor | Raw | CSF | Exceptions | CSF / raw |
|---|---:|---:|---:|---:|
| Qwen L0 E0 `down_proj` scale `[2560, 40]` (slab) | 102 400 B | 54 664 B | 226 (0.22 %) | **53.4 %** |
| GLM L3 E0 `gate_proj` scale `[2048, 256]` (csf32) | 524 288 B | 283 624 B | 1 428 (0.27 %) | **54.1 %** |

Whole models (✅ headers):

| Model | Raw expert scales | CSF expert scales | Saved |
|---|---:|---:|---:|
| Qwen FP4-Local | 7.03 GiB | 3.69 GiB (fixed 3.60 + exceptions 0.09) | **3.3 GiB** |
| GLM FP4-Local | 17.7 GiB | 9.6 GiB (codes 8.86, offsets 0.56, exceptions 0.10, bases 0.09, page bases 0.004) | **8.1 GiB** |

## 5. Who reads CSF today

| Code | Slab `csf` | `csf_32x2_v1` | Where the scales live in VRAM |
|---|---|---|---|
| Upstream vLLM `main` | ❌ | ❌ | (can't load) |
| PR #60646 (draft) | ✅ decoded on the CPU at load | ❌ | raw (CSF saving lost in VRAM) |
| LIL vLLM fork + b12x | ✅ | ✅ | **compressed**; b12x decodes the routed experts' scales into scratch per MoE call (note 14) |

## 6. What this means for Hopper

- **Phase A (load):** decode at load time on the CPU, like PR #60646, and add the `csf32` variant. Simple and safe. GPU memory then holds raw scales: you lose the 3.3 / 8.1 GiB saving, but nothing else changes.
- **Phase C (keep small):** keep the CSF streams in VRAM, and decode one layer's scales (or only the routed experts' scales) into a scratch buffer just before the MoE kernel runs. Details in note 18.

## Key takeaways

- CSF = per-row base byte + 4-bit codes + rare exceptions. Lossless.
- Two variants: slab (2 tensors) and `csf_32x2_v1` (5 tensors, GPU-friendly). Upstream vLLM reads neither; PR #60646 reads the slab one.
- It saves about 46 % of scale memory: 3.3 GiB on Qwen, 8.1 GiB on GLM.

---

**Next:** [12 · Comparing the three quants](12-Comparing-The-Three-Quants.md)
