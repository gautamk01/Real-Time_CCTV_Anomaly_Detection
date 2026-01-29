"""Edge Vision Model - Moondream2 Wrapper."""

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional


class EdgeVision:
    """
    Local vision model for real-time frame analysis.
    Uses Moondream2 for visual question answering.
    """
    
    def __init__(self, model_id: str = "vikhyatk/moondream2", device: str = "auto"):
        """
        Initialize the edge vision model.
        
        Args:
            model_id: HuggingFace model identifier
            device: Device to use ("cuda", "cpu", or "auto")
        """
        self.model_id = model_id
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"🔧 [EDGE] Loading {model_id} on {self.device}...")
        
        # Load model with appropriate dtype
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            local_files_only=True  # ⚡ Force offline mode
        ).to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        
        print(f"✅ [EDGE] Model loaded successfully (Offline Mode)")
    
    def analyze(self, image: Image.Image, question: str) -> str:
        """
        Analyze image and answer question.
        
        Args:
            image: PIL Image in RGB format
            question: Question to ask about the image
            
        Returns:
            Answer string
        """
        try:
            # Encode image
            enc_image = self.model.encode_image(image)
            
            # Get answer
            answer = self.model.answer_question(enc_image, question, self.tokenizer)
            
            return answer.strip()
        
        except Exception as e:
            print(f"   ❌ [EDGE ERROR] {e}")
            return f"Error: {str(e)}"
    
    def get_device_info(self) -> dict:
        """Get information about the device and model."""
        return {
            "model": self.model_id,
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "dtype": str(self.model.dtype)
        }
