"""Edge Vision Model - Moondream2 Wrapper with Latency-Oriented Optimizations."""

import gc
import hashlib
import os
import sys
import traceback
from typing import Optional

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# CUDA optimizations - safe defaults for inference on the edge GPU.
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class EdgeVision:
    """Local vision model for real-time frame analysis."""

    TARGET_SIZE = (378, 378)

    DEFAULT_ANALYZE_MAX_TOKENS = 24
    DEFAULT_FOLLOWUP_MAX_TOKENS = 16

    AUTO_QUANT_FALLBACKS = ("8bit", "4bit")
    SUPPORTED_QUANT_MODES = {"auto", "8bit", "4bit", "none"}
    MIN_AUTO_HEADROOM_GB = 1.0

    def __init__(
        self,
        model_id: str = "vikhyatk/moondream2",
        device: str = "auto",
        quant_mode: str = "auto",
    ):
        self.model_id = model_id

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.requested_quant_mode = self._normalize_quant_mode(quant_mode)
        self.loaded_quant_mode = "none"
        self._enc_cache_key: Optional[str] = None
        self._enc_cache_val = None

        print(f"🔧 [EDGE] Loading {model_id} on {self.device}...")
        print(f"   ⚙️  Requested quantization: {self.requested_quant_mode}")

        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        self.model = self._load_model_with_fallbacks()
        self.model.eval()

        if self.loaded_quant_mode in {"8bit", "4bit"}:
            self._patch_linear_for_bnb()

        self._apply_single_crop_optimization()

        print("✅ [EDGE] Model loaded successfully")
        print(f"   📐 Target resize: {self.TARGET_SIZE}")
        print(
            "   🎯 Token budgets: "
            f"initial={self.DEFAULT_ANALYZE_MAX_TOKENS}, "
            f"follow-up={self.DEFAULT_FOLLOWUP_MAX_TOKENS}"
        )
        print(f"   🧮 Loaded quantization: {self.loaded_quant_mode}")
        print("   🚀 Single-crop ViT optimization: enabled")

        if self.device == "cuda":
            free_mem = (
                torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated()
            )
            print(f"   🎮 GPU memory free: {free_mem / 1024**3:.2f} GB")

    def _normalize_quant_mode(self, quant_mode: str) -> str:
        normalized = str(quant_mode or "auto").strip().lower().replace("-", "")
        aliases = {"8": "8bit", "4": "4bit", "fp": "none", "full": "none"}
        normalized = aliases.get(normalized, normalized)
        if normalized not in self.SUPPORTED_QUANT_MODES:
            raise ValueError(
                f"Unsupported quant_mode={quant_mode!r}. "
                f"Expected one of {sorted(self.SUPPORTED_QUANT_MODES)}."
            )
        return normalized

    def _candidate_quant_modes(self) -> list[str]:
        if self.device != "cuda":
            return ["none"]
        if self.requested_quant_mode == "auto":
            return list(self.AUTO_QUANT_FALLBACKS)
        return [self.requested_quant_mode]

    def _load_kwargs_for_quant_mode(self, quant_mode: str) -> dict:
        kwargs = {
            "trust_remote_code": True,
            "local_files_only": True,
            "device_map": self.device,
        }

        if quant_mode == "8bit":
            kwargs["dtype"] = torch.float16
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif quant_mode == "4bit":
            kwargs["dtype"] = torch.float16
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            kwargs["dtype"] = torch.bfloat16 if self.device == "cuda" else torch.float32

        return kwargs

    def _load_model_with_fallbacks(self):
        errors = []

        for quant_mode in self._candidate_quant_modes():
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    **self._load_kwargs_for_quant_mode(quant_mode),
                )

                if (
                    self.requested_quant_mode == "auto"
                    and quant_mode == "8bit"
                    and not self._has_sufficient_auto_headroom()
                ):
                    free_gb = self._free_gpu_gb()
                    errors.append(
                        f"8bit left only {free_gb:.2f} GB free; trying 4bit fallback"
                    )
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue

                self.loaded_quant_mode = quant_mode
                return model
            except Exception as exc:
                errors.append(f"{quant_mode}: {exc}")
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()

        raise RuntimeError(
            "Failed to load edge model. Tried "
            f"{self._candidate_quant_modes()} with errors: {errors}"
        )

    def _free_gpu_gb(self) -> float:
        if self.device != "cuda":
            return 0.0
        props = torch.cuda.get_device_properties(0)
        free_mem = props.total_memory - torch.cuda.memory_allocated()
        return free_mem / 1024**3

    def _has_sufficient_auto_headroom(self) -> bool:
        if self.device != "cuda":
            return True
        return self._free_gpu_gb() >= self.MIN_AUTO_HEADROOM_GB

    def _patch_linear_for_bnb(self):
        """Patch Moondream helper functions so bitsandbytes modules work correctly."""
        import torch.nn.functional as _F

        inner_model = self.model.model
        moondream_mod = sys.modules[type(inner_model).__module__]
        package = moondream_mod.__name__.rsplit(".", 1)[0]
        layers_mod = sys.modules[package + ".layers"]
        vision_mod = sys.modules[package + ".vision"]
        text_mod = sys.modules[package + ".text"]

        def _bnb_safe_linear(x, w):
            if isinstance(w, torch.nn.Module):
                return w(x)
            return _F.linear(x, w.weight, w.bias)

        def _mixed_dtype_layer_norm(x, w):
            input_dtype = x.dtype
            return _F.layer_norm(
                x.to(w.weight.dtype), w.bias.shape, w.weight, w.bias
            ).to(input_dtype)

        layers_mod.linear = _bnb_safe_linear
        layers_mod.layer_norm = _mixed_dtype_layer_norm
        vision_mod.layer_norm = _mixed_dtype_layer_norm
        text_mod.layer_norm = _mixed_dtype_layer_norm

    def _apply_single_crop_optimization(self):
        """Skip redundant local-crop processing for <= 378x378 inputs."""
        inner_model = self.model.model
        moondream_mod = sys.modules[type(inner_model).__module__]
        _prepare_crops = moondream_mod.prepare_crops
        _reconstruct = moondream_mod.reconstruct_from_crops

        def _fast_vision_encoder(image):
            config = inner_model.config.vision
            all_crops, tiling = _prepare_crops(image, config, device=inner_model.device)

            if tiling == (1, 1):
                global_crop = all_crops[0:1]
                features = inner_model._vis_enc(global_crop)[0]
                reconstructed = features.view(
                    config.enc_n_layers, config.enc_n_layers, config.enc_dim
                )
                img_emb = inner_model._vis_proj(features, reconstructed)
                return img_emb.to(inner_model.text.wte.dtype)

            torch._dynamo.mark_dynamic(all_crops, 0)
            outputs = inner_model._vis_enc(all_crops)
            global_features = outputs[0]
            local_features = outputs[1:].view(
                -1, config.enc_n_layers, config.enc_n_layers, config.enc_dim
            )
            reconstructed = _reconstruct(
                local_features,
                tiling,
                patch_size=1,
                overlap_margin=config.overlap_margin,
            )
            img_emb = inner_model._vis_proj(global_features, reconstructed)
            return img_emb.to(inner_model.text.wte.dtype)

        inner_model._run_vision_encoder = _fast_vision_encoder

    def _resize_image(self, image: Image.Image) -> Image.Image:
        if image.size != self.TARGET_SIZE:
            return image.resize(self.TARGET_SIZE, Image.LANCZOS)
        return image

    def _image_cache_key(self, image: Image.Image) -> str:
        digest = hashlib.blake2b(image.tobytes(), digest_size=16).hexdigest()
        return f"{image.width}x{image.height}:{digest}"

    def _query_settings(self, max_tokens: int) -> dict:
        return {
            "max_tokens": max(1, int(max_tokens)),
            "temperature": 0,
        }

    def encode(self, image: Image.Image):
        """Phase 1: Encode image into embeddings."""
        resized = self._resize_image(image)
        cache_key = self._image_cache_key(resized)

        if self._enc_cache_key == cache_key and self._enc_cache_val is not None:
            return self._enc_cache_val

        with torch.inference_mode():
            enc_image = self.model.encode_image(resized)

        self._enc_cache_key = cache_key
        self._enc_cache_val = enc_image
        return enc_image

    def answer(
        self,
        encoded_image,
        question: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Phase 2: Answer a question using a pre-encoded image."""
        token_budget = (
            self.DEFAULT_FOLLOWUP_MAX_TOKENS
            if max_tokens is None
            else int(max_tokens)
        )
        with torch.inference_mode():
            result = self.model.query(
                encoded_image,
                question,
                settings=self._query_settings(token_budget),
            )
        return result["answer"].strip()

    def analyze(
        self,
        image: Image.Image,
        question: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Combined encode + answer path."""
        token_budget = (
            self.DEFAULT_ANALYZE_MAX_TOKENS
            if max_tokens is None
            else int(max_tokens)
        )

        try:
            enc_image = self.encode(image)
            return self.answer(enc_image, question, max_tokens=token_budget)
        except torch.cuda.OutOfMemoryError:
            print("   ⚠️  [EDGE] GPU OOM - clearing cache and retrying once...")
            self.clear_cache()
            try:
                enc_image = self.encode(image)
                return self.answer(enc_image, question, max_tokens=token_budget)
            except Exception as exc:
                return f"Error (OOM recovery failed): {exc}"
        except Exception as exc:
            print(f"   ❌ [EDGE ERROR] {exc}")
            traceback.print_exc()
            return f"Error: {exc}"

    def clear_cache(self):
        """Clear GPU memory and the encoded-image cache."""
        self._enc_cache_key = None
        self._enc_cache_val = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def get_device_info(self) -> dict:
        gpu_mem = {}
        if self.device == "cuda":
            props = torch.cuda.get_device_properties(0)
            gpu_mem = {
                "total": f"{props.total_memory / 1024**3:.2f} GB",
                "allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
                "free": f"{self._free_gpu_gb():.2f} GB",
            }

        return {
            "model": self.model_id,
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "dtype": str(self.model.dtype),
            "quant_mode_requested": self.requested_quant_mode,
            "quant_mode_loaded": self.loaded_quant_mode,
            "target_size": self.TARGET_SIZE,
            "default_analyze_max_tokens": self.DEFAULT_ANALYZE_MAX_TOKENS,
            "default_followup_max_tokens": self.DEFAULT_FOLLOWUP_MAX_TOKENS,
            "gpu_memory": gpu_mem,
        }
