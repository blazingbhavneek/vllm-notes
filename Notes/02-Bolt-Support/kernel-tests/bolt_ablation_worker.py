"""Benchmark-only worker variants; no changes to production backend selection."""

import torch
import torch.nn.functional as F
from vllm.v1.worker.gpu_worker import Worker


class BaselineWorker(Worker):
    def load_model(self, **kwargs):
        from bolt_baseline_experts import ModelOptBlockTritonExperts as Original
        import bolt_baseline_warmup
        from vllm.model_executor.layers.fused_moe.experts import (
            modelopt_block_triton_moe as mod,
        )
        from vllm.model_executor.warmup import modelopt_block_triton_warmup as warm
        import bolt_baseline_experts
        from vllm.model_executor.layers.quantization import (
            modelopt_block_quant_triton as kernels,
        )

        old_config = kernels.moe_config

        def config(*args):
            cfg = {**old_config(*args), "SINGLETON_GEMV": False}
            if cfg["BLOCK_SIZE_M"] == 64 and cfg["BN"] == 32:
                cfg["BK"] = 256
            return cfg

        kernels.moe_config = config
        bolt_baseline_experts.moe_config = config
        mod.ModelOptBlockTritonExperts.apply = Original.apply
        warm._warm_moe = bolt_baseline_warmup._warm_moe
        return super().load_model(**kwargs)


class DenseCacheWorker(Worker):
    def load_model(self, **kwargs):
        from vllm.model_executor.layers.quantization.modelopt_block_quant import (
            ModelOptBlockQuantLinearMethod as Method,
        )
        from vllm.model_executor.layers.quantization.modelopt_block_quant_triton import (
            dequantize_blocks,
        )

        original = Method.apply

        def cached_apply(method, layer, x, bias=None):
            if hasattr(layer, "dense_bf16_weight"):
                return F.linear(x, layer.dense_bf16_weight, bias)
            return original(method, layer, x, bias)

        Method.apply = cached_apply
        super().load_model(**kwargs)
        count, total = 0, 0
        with torch.inference_mode():
            for layer in self.get_model().modules():
                method = getattr(layer, "quant_method", None)
                if (
                    isinstance(method, Method)
                    and method.use_triton
                    and method.codec == "iq2_xxs"
                ):
                    weight = dequantize_blocks(
                        layer.weight, layer.weight_scale, layer.iq2_lut, method.codec
                    )
                    layer.register_buffer("dense_bf16_weight", weight, persistent=False)
                    count += 1
                    total += weight.numel() * weight.element_size()
        self.model_runner.model_memory_usage += total
        print("DENSE_CACHE", count, total, flush=True)


class GemvGateWorker(Worker):
    def load_model(self, **kwargs):
        from vllm.model_executor.layers.fused_moe.experts import (
            modelopt_block_triton_moe as mod,
        )
        from vllm.model_executor.layers.quantization import (
            modelopt_block_quant_triton as kernels,
        )

        limit = int(self.vllm_config.additional_config["gemv_max_tokens"])

        def gate(M, E, top_k, codec):
            if M > limit:
                return None
            bn, bk, w = (8, 256, 1) if codec == "iq2_xxs" else (32, 256, 4)
            return {"BN": bn, "BK": bk, "num_warps": w, "num_stages": 2}

        mod.moe_gemv_config = gate
        kernels.moe_gemv_config = gate
        return super().load_model(**kwargs)
