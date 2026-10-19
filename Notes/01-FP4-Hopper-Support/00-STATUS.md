# Status of these notes (19 Oct 2026)

**All notes 01–23 are written.** Start at [01 · Start here](01-Start-Here.md).

- For understanding: read 01 → 19 in order.
- For implementing: notes 20 (tasks T0–T12), 21 (tests and commands), 22 (glossary) and 23 (open questions, risks, people).

## What was checked, and how

- ✅ Checked from code (vLLM `main` at `3e96f9cf63`), checkpoint headers, and a few small tensors (about 6 MB, fetched with HTTP range requests). No model weights or containers were downloaded.
- ✅ The scripts in `scripts/` were run, and their outputs match the notes. Note 21, section 8, explains how to rerun every one.
  - Both CSF decoders give back the exact original scales on real tensors.
  - The encoder rule in note 11, section 1, reproduces NVIDIA's bytes exactly for both formats.
- ⚠️ **Nothing was run on a GPU.** No vLLM server and no Hopper test. Task T0 in note 20 is the first step for that.

## Scripts

| Script | What it does |
|---|---|
| `st_header.py`, `st_tensor.py`, `st_slice.py` | fetch a safetensors header / one tensor / the first bytes of a tensor |
| `fp4_scale_exactness.py` | when an FP4 × E4M3 product is exact in E4M3 |
| `csf_slab_decode.py` | slab CSF decode + re-encode check (Qwen) |
| `csf32_decode.py` | `csf_32x2_v1` loop decoder (GLM), the simplest reference |
| `csf32_vectorized.py` | vectorized decoder + test-only encoder; port this one to vLLM (task T3) |
| `w4a8_error_study.py` | error of fused FP8 weights and FP8 activations on one real expert |
| `kld_topk.py` | KL divergence between two vLLM runs (the dump part has not been run yet) |
