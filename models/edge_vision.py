"""Edge Vision Model - Moondream2 Wrapper with Speed Optimizations."""

import sys
import torch
import gc
import hashlib
import traceback
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

# CUDA Optimizations - Memory friendly settings
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class EdgeVision:
    """
    Local vision model for real-time frame analysis.
    Uses Moondream2 for visual question answering.

    Speed Optimizations Applied:
    - Native 378x378 resolution (matches Moondream2 crop_size, avoids blurry upscale)
    - Single-crop ViT: skips redundant local crop for ≤crop_size images (~50% ViT speedup)
    - Native bfloat16 dtype (eliminates float16↔bfloat16 casting overhead)
    - CUDA/cuDNN optimizations (TF32, cuDNN benchmark)
    - Content-hash image cache (skips encode_image on same frame)
    - Direct model.query() API with enforced max_tokens and greedy decoding
    - Two-phase API (encode/answer) for pipeline parallelism
    """

    # Native Moondream2 crop size — avoids internal upscale from 224→378
    TARGET_SIZE = (378, 378)

    # Max tokens for Edge responses — enforced via model.query() settings.
    # Moondream2 typically stops at ~25-30 words (EOS) for scene descriptions.
    # 40 tokens provides a safety margin without wasting compute on runaway generation.
    MAX_NEW_TOKENS = 76

    def __init__(self, model_id: str = "vikhyatk/moondream2", device: str = "auto"):
        """
        Initialize the edge vision model.

        Args:
            model_id: HuggingFace model identifier
            device: Device to use ("cuda", "cpu", or "auto")
        """
        self.model_id = model_id

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"🔧 [EDGE] Loading {model_id} on {self.device}...")

        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        # Use the model's native bfloat16 dtype — avoids casting overhead
        # since Moondream2's prepare_crops() outputs bfloat16 tensors.
        # bfloat16 and float16 use identical memory (2 bytes/param).
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            local_files_only=True
        ).to(self.device)

        # Set eval mode — disables dropout, correct inference behavior
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

        # Apply single-crop optimization (skip redundant local crop for ≤crop_size images)
        self._apply_single_crop_optimization()

        # Sampling settings: greedy decoding (temperature=0) is faster and
        # more deterministic than the default (temp=0.5, top_p=0.3)
        self._query_settings = {
            "max_tokens": self.MAX_NEW_TOKENS,
            "temperature": 0,
        }

        # Content-hash image encoding cache (avoids re-encoding same frame)
        self._enc_cache_hash = None
        self._enc_cache_val = None

        print(f"✅ [EDGE] Model loaded successfully (Optimized Mode)")
        print(f"   📐 Target resize: {self.TARGET_SIZE}")
        print(f"   🎯 Max tokens: {self.MAX_NEW_TOKENS} (enforced), greedy decoding")
        print(f"   🚀 Single-crop ViT optimization: enabled")

        if self.device == "cuda":
            free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            print(f"   🎮 GPU memory free: {free_mem / 1024**3:.2f} GB")

    def _apply_single_crop_optimization(self):
        """
        Monkey-patch the model's vision encoder to skip redundant crop processing.

        For images ≤ crop_size (378x378), Moondream2 creates 2 identical crops
        (global + local) and processes both through the 27-layer ViT. Since both
        crops produce the same features, the second is pure wasted computation.

        This patch processes only 1 crop and reuses its features for both paths
        in vision_projection, cutting ViT forward-pass time roughly in half.
        """
        # HfMoondream.encode_image is a property that delegates to
        # self.model.encode_image (the inner MoondreamModel). We must patch
        # the inner model's _run_vision_encoder, not the wrapper's.
        inner_model = self.model.model  # MoondreamModel instance

        # Access prepare_crops from the moondream module (uses relative imports,
        # so we can't import it directly — grab from the loaded module instead)
        moondream_mod = sys.modules[type(inner_model).__module__]
        _prepare_crops = moondream_mod.prepare_crops
        _reconstruct = moondream_mod.reconstruct_from_crops

        def _fast_vision_encoder(image):
            config = inner_model.config.vision
            all_crops, tiling = _prepare_crops(image, config, device=inner_model.device)

            if tiling == (1, 1):
                # Both crops are identical for images ≤ crop_size.
                # Process only the global crop (index 0) through the ViT
                # and reuse the same features for both global and local paths.
                global_crop = all_crops[0:1]  # [1, C, 378, 378]
                features = inner_model._vis_enc(global_crop)[0]  # [729, enc_dim]
                reconstructed = features.view(
                    config.enc_n_layers, config.enc_n_layers, config.enc_dim
                )
                return inner_model._vis_proj(features, reconstructed)
            else:
                # Original path for larger images with multiple crops
                torch._dynamo.mark_dynamic(all_crops, 0)
                outputs = inner_model._vis_enc(all_crops)
                global_features = outputs[0]
                local_features = outputs[1:].view(
                    -1, config.enc_n_layers, config.enc_n_layers, config.enc_dim
                )
                reconstructed = _reconstruct(
                    local_features, tiling,
                    patch_size=1, overlap_margin=config.overlap_margin,
                )
                return inner_model._vis_proj(global_features, reconstructed)

        # Patch the inner MoondreamModel instance (not the HfMoondream wrapper).
        inner_model._run_vision_encoder = _fast_vision_encoder

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image to native Moondream2 resolution."""
        if image.size != self.TARGET_SIZE:
            return image.resize(self.TARGET_SIZE, Image.LANCZOS)
        return image

    def _image_hash(self, image: Image.Image) -> str:
        """Fast content hash for cache keying (~0.1ms for 378x378)."""
        return hashlib.md5(image.tobytes()).hexdigest()

    def encode(self, image: Image.Image):
        """
        Phase 1: Encode image into embeddings (heavy GPU work).

        This is the expensive step (~200-500ms). Use this for pipeline
        parallelism — start encoding the next frame while the cloud API
        processes the current round.

        Args:
            image: PIL Image in RGB format

        Returns:
            EncodedImage object (opaque, pass to answer())
        """
        resized = self._resize_image(image)
        img_hash = self._image_hash(image)

        with torch.inference_mode():
            if img_hash == self._enc_cache_hash and self._enc_cache_val is not None:
                return self._enc_cache_val

            enc_image = self.model.encode_image(resized)
            self._enc_cache_hash = img_hash
            self._enc_cache_val = enc_image
            return enc_image

    def answer(self, encoded_image, question: str) -> str:
        """
        Phase 2: Answer question using pre-encoded image (lighter step).

        Args:
            encoded_image: Result from encode()
            question: Question to ask about the image

        Returns:
            Answer string
        """
        with torch.inference_mode():
            result = self.model.query(
                encoded_image, question, settings=self._query_settings
            )
            return result["answer"].strip()

    def analyze(self, image: Image.Image, question: str) -> str:
        """
        Analyze image and answer question (combined encode + answer).

        Args:
            image: PIL Image in RGB format
            question: Question to ask about the image

        Returns:
            Answer string
        """
        try:
            enc_image = self.encode(image)
            return self.answer(enc_image, question)

        except torch.cuda.OutOfMemoryError:
            print(f"   ⚠️  [EDGE] GPU OOM - clearing cache and retrying...")
            self.clear_cache()
            torch.cuda.empty_cache()
            gc.collect()

            try:
                resized = self._resize_image(image)
                with torch.inference_mode():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        enc_image = self.model.encode_image(resized)
                        result = self.model.query(
                            enc_image, question, settings=self._query_settings
                        )
                return result["answer"].strip()
            except Exception as e:
                return f"Error (OOM recovery failed): {str(e)}"

        except Exception as e:
            print(f"   ❌ [EDGE ERROR] {e}")
            traceback.print_exc()
            return f"Error: {str(e)}"

    def clear_cache(self):
        """Clear GPU memory and encoding cache."""
        self._enc_cache_hash = None
        self._enc_cache_val = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    def get_device_info(self) -> dict:
        """Get information about the device and model."""
        gpu_mem = {}
        if self.device == "cuda":
            gpu_mem = {
                "total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB",
                "allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
                "free": f"{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3:.2f} GB"
            }

        return {
            "model": self.model_id,
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "dtype": str(self.model.dtype),
            "target_size": self.TARGET_SIZE,
            "max_tokens": self.MAX_NEW_TOKENS,
            "gpu_memory": gpu_mem
        }
