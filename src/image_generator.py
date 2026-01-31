"""
Image Generator Module - AI Image Generation
=============================================
Generates polished images from sketches using AI models.
Supports multiple backends: Hugging Face Inference API with fal.ai provider,
ControlNet for sketch conditioning, and local diffusers.

Recommended Models for Sketch-to-Image:
- ControlNet Scribble: Best for hand-drawn doodles
- ControlNet Canny: Best for edge-based sketches  
- SDXL + ControlNet: Highest quality output
"""

import os
import numpy as np
from typing import Optional, Tuple, Dict, Any, Callable
from PIL import Image
import io
import base64
import time
import threading
from dataclasses import dataclass
from enum import Enum, auto
import json


class GeneratorBackend(Enum):
    """Available image generation backends."""
    HUGGINGFACE_FAL = auto()         # Hugging Face with fal.ai provider (recommended)
    HUGGINGFACE_INFERENCE = auto()   # Hugging Face Inference API
    LOCAL_DIFFUSERS = auto()         # Local diffusers library
    REPLICATE = auto()               # Replicate API


@dataclass
class GenerationRequest:
    """Request for image generation."""
    sketch: np.ndarray
    prompt: str
    negative_prompt: str = ""
    strength: float = 0.75  # How much to transform the sketch
    guidance_scale: float = 7.5
    num_inference_steps: int = 30
    seed: Optional[int] = None
    style: str = "digital_art"


@dataclass 
class GenerationResult:
    """Result of image generation."""
    success: bool
    image: Optional[np.ndarray]
    error: Optional[str]
    generation_time: float
    metadata: Dict[str, Any]


class StylePresets:
    """Predefined style presets for image generation."""
    
    PRESETS = {
        'digital_art': {
            'prompt_suffix': ', digital art, clean lines, vibrant colors, professional illustration',
            'negative_prompt': 'blurry, low quality, distorted, ugly, bad anatomy',
            'guidance_scale': 7.5,
            'strength': 0.75
        },
        'anime': {
            'prompt_suffix': ', anime style, cel shaded, vibrant colors, detailed, studio ghibli inspired',
            'negative_prompt': 'realistic, photograph, 3d render, blurry, bad quality',
            'guidance_scale': 8.0,
            'strength': 0.8
        },
        'realistic': {
            'prompt_suffix': ', photorealistic, highly detailed, 8k, professional photography',
            'negative_prompt': 'cartoon, anime, drawing, illustration, painting, blurry',
            'guidance_scale': 7.0,
            'strength': 0.7
        },
        'cartoon': {
            'prompt_suffix': ', cartoon style, colorful, fun, playful, pixar style',
            'negative_prompt': 'realistic, photograph, dark, scary, violent',
            'guidance_scale': 7.5,
            'strength': 0.8
        },
        'sketch': {
            'prompt_suffix': ', pencil sketch, detailed line art, artistic, hand drawn',
            'negative_prompt': 'color, painted, digital, 3d, photograph',
            'guidance_scale': 6.5,
            'strength': 0.6
        },
        'watercolor': {
            'prompt_suffix': ', watercolor painting, soft colors, artistic, flowing, traditional art',
            'negative_prompt': 'digital, sharp lines, photograph, 3d render',
            'guidance_scale': 7.0,
            'strength': 0.75
        },
        'oil_painting': {
            'prompt_suffix': ', oil painting, classical art, textured, masterpiece, museum quality',
            'negative_prompt': 'digital, cartoon, anime, photograph, low quality',
            'guidance_scale': 7.5,
            'strength': 0.7
        }
    }
    
    @classmethod
    def get_preset(cls, name: str) -> dict:
        """Get a style preset by name."""
        return cls.PRESETS.get(name, cls.PRESETS['digital_art'])
    
    @classmethod
    def get_all_names(cls) -> list:
        """Get all preset names."""
        return list(cls.PRESETS.keys())


class ImageGenerator:
    """
    AI image generator that transforms sketches into polished images.
    
    Supports multiple backends and style presets.
    Uses ControlNet-based models for best sketch-to-image results.
    """
    
    # Recommended models for sketch-to-image
    SKETCH_MODELS = {
        'controlnet_scribble': 'lllyasviel/control_v11p_sd15_scribble',
        'controlnet_canny': 'lllyasviel/control_v11p_sd15_canny',
        'sdxl_controlnet': 'diffusers/controlnet-canny-sdxl-1.0',
        'img2img': 'stabilityai/stable-diffusion-2-1',
        'hunyuan': 'tencent/HunyuanImage-3.0-Instruct',  # For image editing
    }
    
    def __init__(
        self,
        backend: GeneratorBackend = GeneratorBackend.HUGGINGFACE_FAL,
        api_key: Optional[str] = None,
        model_id: str = "black-forest-labs/FLUX.1-schnell"  # Fast, good quality
    ):
        """
        Initialize the image generator.
        
        Args:
            backend: Which backend to use for generation
            api_key: API key (HF_TOKEN for Hugging Face)
            model_id: Model identifier for the backend
            
        Recommended models:
            - black-forest-labs/FLUX.1-schnell: Fast, good quality (fal.ai)
            - stabilityai/stable-diffusion-xl-base-1.0: High quality
            - runwayml/stable-diffusion-v1-5: Classic, reliable
        """
        self.backend = backend
        self.api_key = api_key or os.environ.get('HF_TOKEN', '') or os.environ.get('HF_API_KEY', '')
        self.model_id = model_id
        
        # Generation state
        self._is_generating = False
        self._generation_thread: Optional[threading.Thread] = None
        self._last_result: Optional[GenerationResult] = None
        
        # Callbacks
        self._on_complete: Optional[Callable[[GenerationResult], None]] = None
        self._on_progress: Optional[Callable[[float, str], None]] = None
        
        # Initialize backend
        self._init_backend()
    
    def _init_backend(self):
        """Initialize the selected backend."""
        if self.backend in [GeneratorBackend.HUGGINGFACE_FAL, GeneratorBackend.HUGGINGFACE_INFERENCE]:
            try:
                # Use huggingface_hub for modern API access
                from huggingface_hub import InferenceClient
                
                if self.backend == GeneratorBackend.HUGGINGFACE_FAL:
                    # fal.ai provider - faster, supports more models
                    self._client = InferenceClient(
                        provider="fal-ai",
                        api_key=self.api_key,
                    )
                    print("[INFO] Using Hugging Face with fal.ai provider")
                else:
                    # Standard HF Inference API
                    self._client = InferenceClient(
                        api_key=self.api_key,
                    )
                    print("[INFO] Using Hugging Face Inference API")
                    
            except ImportError:
                print("[WARNING] huggingface_hub not installed. Install with: pip install huggingface_hub")
                self._client = None
                # Fallback to requests
                try:
                    import requests
                    self._requests = requests
                except ImportError:
                    print("[WARNING] requests not installed either")
        
        elif self.backend == GeneratorBackend.LOCAL_DIFFUSERS:
            try:
                from diffusers import StableDiffusionImg2ImgPipeline
                import torch
                
                self._pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                
                if torch.cuda.is_available():
                    self._pipe = self._pipe.to("cuda")
                    print("[INFO] Using CUDA for image generation")
                else:
                    print("[INFO] Using CPU for image generation (this will be slow)")
                    
            except ImportError:
                print("[WARNING] diffusers/torch not installed. Install with: pip install diffusers torch")
                self._pipe = None
    
    def generate(
        self,
        request: GenerationRequest,
        async_mode: bool = True
    ) -> Optional[GenerationResult]:
        """
        Generate an image from a sketch.
        
        Args:
            request: Generation request with sketch and parameters
            async_mode: If True, run in background thread
            
        Returns:
            GenerationResult if sync mode, None if async mode
        """
        if self._is_generating:
            return GenerationResult(
                success=False,
                image=None,
                error="Generation already in progress",
                generation_time=0,
                metadata={}
            )
        
        if async_mode:
            self._is_generating = True
            self._generation_thread = threading.Thread(
                target=self._generate_async,
                args=(request,),
                daemon=True
            )
            self._generation_thread.start()
            return None
        else:
            return self._generate_sync(request)
    
    def _generate_sync(self, request: GenerationRequest) -> GenerationResult:
        """Synchronous generation."""
        start_time = time.time()
        
        try:
            # Apply style preset
            preset = StylePresets.get_preset(request.style)
            full_prompt = request.prompt + preset['prompt_suffix']
            full_negative = request.negative_prompt + ', ' + preset['negative_prompt']
            
            # Convert sketch to PIL Image
            if len(request.sketch.shape) == 2:
                # Grayscale - convert to RGB
                sketch_rgb = np.stack([request.sketch] * 3, axis=-1)
            else:
                sketch_rgb = request.sketch
            
            pil_sketch = Image.fromarray(sketch_rgb).convert('RGB')
            pil_sketch = pil_sketch.resize((512, 512))
            
            # Generate based on backend
            if self.backend == GeneratorBackend.HUGGINGFACE_FAL:
                result_image = self._generate_hf_fal(
                    pil_sketch, full_prompt, full_negative, request
                )
            elif self.backend == GeneratorBackend.HUGGINGFACE_INFERENCE:
                result_image = self._generate_hf_inference(
                    pil_sketch, full_prompt, full_negative, request
                )
            elif self.backend == GeneratorBackend.LOCAL_DIFFUSERS:
                result_image = self._generate_local(
                    pil_sketch, full_prompt, full_negative, request
                )
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")
            
            generation_time = time.time() - start_time
            
            # Convert to numpy
            result_array = np.array(result_image)
            
            return GenerationResult(
                success=True,
                image=result_array,
                error=None,
                generation_time=generation_time,
                metadata={
                    'prompt': full_prompt,
                    'style': request.style,
                    'backend': self.backend.name
                }
            )
            
        except Exception as e:
            return GenerationResult(
                success=False,
                image=None,
                error=str(e),
                generation_time=time.time() - start_time,
                metadata={}
            )
    
    def _generate_async(self, request: GenerationRequest):
        """Asynchronous generation in background thread."""
        result = self._generate_sync(request)
        self._last_result = result
        self._is_generating = False
        
        if self._on_complete:
            self._on_complete(result)
    
    def _generate_hf_fal(
        self,
        sketch: Image.Image,
        prompt: str,
        negative_prompt: str,
        request: GenerationRequest
    ) -> Image.Image:
        """
        Generate using Hugging Face with fal.ai provider.
        Best for sketch-to-image with image_to_image endpoint.
        """
        if self._client is None:
            raise ValueError("Hugging Face client not initialized. Check HF_TOKEN.")
        
        # Convert sketch to bytes
        buffer = io.BytesIO()
        sketch.save(buffer, format='PNG')
        sketch_bytes = buffer.getvalue()
        
        # Use image_to_image for sketch transformation
        # This preserves the sketch structure while generating details
        result_image = self._client.image_to_image(
            image=sketch_bytes,
            prompt=prompt,
            model=self.model_id,
            negative_prompt=negative_prompt,
            guidance_scale=request.guidance_scale,
            strength=request.strength,  # How much to transform (0.7-0.85 good for sketches)
        )
        
        return result_image
    
    def _generate_hf_inference(
        self,
        sketch: Image.Image,
        prompt: str,
        negative_prompt: str,
        request: GenerationRequest
    ) -> Image.Image:
        """Generate using Hugging Face Inference API (standard)."""
        
        # Try modern huggingface_hub client first
        if hasattr(self, '_client') and self._client is not None:
            buffer = io.BytesIO()
            sketch.save(buffer, format='PNG')
            sketch_bytes = buffer.getvalue()
            
            try:
                # Try image_to_image first (best for sketches)
                result_image = self._client.image_to_image(
                    image=sketch_bytes,
                    prompt=prompt,
                    model=self.model_id,
                    negative_prompt=negative_prompt,
                    guidance_scale=request.guidance_scale,
                    strength=request.strength,
                )
                return result_image
            except Exception as e:
                print(f"[INFO] image_to_image failed, trying text_to_image: {e}")
                # Fallback to text_to_image
                result_image = self._client.text_to_image(
                    prompt=prompt,
                    model=self.model_id,
                    negative_prompt=negative_prompt,
                    guidance_scale=request.guidance_scale,
                )
                return result_image
        
        # Fallback to requests-based API
        if not hasattr(self, '_requests'):
            raise ValueError("No HTTP client available")
        
        if not self.api_key:
            raise ValueError("HF_TOKEN environment variable not set")
        
        api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "negative_prompt": negative_prompt,
                "guidance_scale": request.guidance_scale,
                "num_inference_steps": request.num_inference_steps,
            }
        }
        
        if request.seed is not None:
            payload["parameters"]["seed"] = request.seed
        
        response = self._requests.post(api_url, headers=headers, json=payload, timeout=120)
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} - {response.text}")
        
        result_image = Image.open(io.BytesIO(response.content))
        return result_image
    
    def _generate_local(
        self,
        sketch: Image.Image,
        prompt: str,
        negative_prompt: str,
        request: GenerationRequest
    ) -> Image.Image:
        """Generate using local diffusers pipeline."""
        if self._pipe is None:
            raise ValueError("Local diffusers pipeline not initialized")
        
        import torch
        
        generator = None
        if request.seed is not None:
            generator = torch.Generator().manual_seed(request.seed)
        
        result = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=sketch,
            strength=request.strength,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            generator=generator
        )
        
        return result.images[0]
    
    def set_on_complete(self, callback: Callable[[GenerationResult], None]):
        """Set callback for when generation completes."""
        self._on_complete = callback
    
    def set_on_progress(self, callback: Callable[[float, str], None]):
        """Set callback for progress updates."""
        self._on_progress = callback
    
    def is_generating(self) -> bool:
        """Check if generation is in progress."""
        return self._is_generating
    
    def get_last_result(self) -> Optional[GenerationResult]:
        """Get the last generation result."""
        return self._last_result
    
    def cancel(self):
        """Cancel ongoing generation (best effort)."""
        # Note: Actual cancellation depends on backend support
        self._is_generating = False


class MockImageGenerator(ImageGenerator):
    """
    Mock image generator for testing without API access.
    Creates a stylized version of the sketch.
    """
    
    def __init__(self):
        """Initialize mock generator."""
        self.backend = GeneratorBackend.HUGGINGFACE_INFERENCE
        self._is_generating = False
        self._last_result = None
        self._on_complete = None
        self._on_progress = None
    
    def _generate_sync(self, request: GenerationRequest) -> GenerationResult:
        """Generate a mock result by stylizing the sketch."""
        import cv2
        
        start_time = time.time()
        
        try:
            sketch = request.sketch.copy()
            
            # Simulate processing time
            time.sleep(1.5)
            
            # Create a stylized version of the sketch
            if len(sketch.shape) == 2:
                # Convert grayscale to color
                colored = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
            else:
                colored = sketch.copy()
            
            # Apply style based on preset
            style = request.style
            
            if style == 'anime':
                # Anime-like colors
                colored = cv2.applyColorMap(
                    cv2.cvtColor(colored, cv2.COLOR_BGR2GRAY),
                    cv2.COLORMAP_COOL
                )
            elif style == 'realistic':
                # Sepia tone
                kernel = np.array([[0.272, 0.534, 0.131],
                                   [0.349, 0.686, 0.168],
                                   [0.393, 0.769, 0.189]])
                colored = cv2.transform(colored, kernel)
            elif style == 'watercolor':
                # Soft blur effect
                colored = cv2.bilateralFilter(colored, 15, 75, 75)
                colored = cv2.applyColorMap(
                    cv2.cvtColor(colored, cv2.COLOR_BGR2GRAY),
                    cv2.COLORMAP_OCEAN
                )
            elif style == 'cartoon':
                # Edge-preserving filter
                colored = cv2.edgePreservingFilter(colored, flags=2, sigma_s=50, sigma_r=0.4)
                colored = cv2.applyColorMap(
                    cv2.cvtColor(colored, cv2.COLOR_BGR2GRAY),
                    cv2.COLORMAP_RAINBOW
                )
            else:
                # Default digital art - apply a nice color map
                colored = cv2.applyColorMap(
                    cv2.cvtColor(colored, cv2.COLOR_BGR2GRAY),
                    cv2.COLORMAP_VIRIDIS
                )
            
            # Resize to standard output
            result = cv2.resize(colored, (512, 512))
            
            # Add some enhancement
            result = cv2.convertScaleAbs(result, alpha=1.2, beta=10)
            
            return GenerationResult(
                success=True,
                image=result,
                error=None,
                generation_time=time.time() - start_time,
                metadata={
                    'prompt': request.prompt,
                    'style': request.style,
                    'backend': 'mock',
                    'note': 'This is a mock result. Set HF_API_KEY for real generation.'
                }
            )
            
        except Exception as e:
            return GenerationResult(
                success=False,
                image=None,
                error=str(e),
                generation_time=time.time() - start_time,
                metadata={}
            )


def create_generator(
    use_mock: bool = False,
    api_key: Optional[str] = None,
    use_fal: bool = True
) -> ImageGenerator:
    """
    Factory function to create an image generator.
    
    Args:
        use_mock: If True, use mock generator for testing
        api_key: API key for real generation (HF_TOKEN)
        use_fal: If True, use fal.ai provider (faster, recommended)
        
    Returns:
        ImageGenerator instance
        
    Environment Variables:
        HF_TOKEN: Hugging Face API token (preferred)
        HF_API_KEY: Alternative name for HF token
    """
    if use_mock:
        return MockImageGenerator()
    
    # Check for API key
    key = api_key or os.environ.get('HF_TOKEN', '') or os.environ.get('HF_API_KEY', '')
    
    if not key:
        print("[INFO] No HF_TOKEN found. Using mock generator.")
        print("[INFO] Set HF_TOKEN environment variable for real AI generation.")
        print("[INFO] Get your free token at: https://huggingface.co/settings/tokens")
        return MockImageGenerator()
    
    # Choose backend
    backend = GeneratorBackend.HUGGINGFACE_FAL if use_fal else GeneratorBackend.HUGGINGFACE_INFERENCE
    
    # Recommended models for sketch-to-image:
    # - "black-forest-labs/FLUX.1-schnell" - Fast, good quality
    # - "stabilityai/stable-diffusion-xl-base-1.0" - High quality
    # - "tencent/HunyuanImage-3.0-Instruct" - Good for image editing
    
    return ImageGenerator(
        backend=backend,
        api_key=key,
        model_id="black-forest-labs/FLUX.1-schnell"  # Fast & good for sketches
    )


if __name__ == "__main__":
    # Test image generator
    import cv2
    
    print("Testing Image Generator Module")
    print("=" * 40)
    
    # Create test sketch
    sketch = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(sketch, (256, 200), 80, 255, 3)
    cv2.ellipse(sketch, (256, 350), (100, 60), 0, 0, 180, 255, 3)
    cv2.line(sketch, (200, 280), (200, 350), 255, 3)
    cv2.line(sketch, (312, 280), (312, 350), 255, 3)
    
    # Create generator (will use mock if no API key)
    generator = create_generator()
    
    # Test each style
    for style in ['digital_art', 'anime', 'cartoon', 'watercolor']:
        print(f"\nGenerating {style} style...")
        
        request = GenerationRequest(
            sketch=sketch,
            prompt="A friendly robot character",
            style=style
        )
        
        result = generator.generate(request, async_mode=False)
        
        if result.success:
            print(f"  Success! Time: {result.generation_time:.2f}s")
            cv2.imshow(f"Result - {style}", result.image)
        else:
            print(f"  Failed: {result.error}")
    
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
