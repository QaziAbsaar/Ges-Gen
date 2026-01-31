"""
Sketch Processor Module - Clean & Preprocess Sketch
====================================================
Processes raw sketches into clean images suitable for AI generation.
Includes noise reduction, edge enhancement, and format conversion.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from PIL import Image
import io
import base64


class SketchProcessor:
    """
    Processes raw sketches for AI image generation.
    
    Applies various image processing techniques to clean up
    hand-drawn sketches and prepare them for AI models.
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        invert: bool = True,
        denoise: bool = True,
        enhance_edges: bool = True
    ):
        """
        Initialize the sketch processor.
        
        Args:
            target_size: Output image size (width, height)
            invert: Whether to invert colors (white lines on black -> black on white)
            denoise: Whether to apply denoising
            enhance_edges: Whether to enhance edges
        """
        self.target_size = target_size
        self.invert = invert
        self.denoise = denoise
        self.enhance_edges = enhance_edges
    
    def process(self, sketch: np.ndarray) -> np.ndarray:
        """
        Process a raw sketch image.
        
        Args:
            sketch: Input sketch (grayscale or BGR)
            
        Returns:
            Processed sketch as grayscale image
        """
        # Convert to grayscale if needed
        if len(sketch.shape) == 3:
            processed = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
        else:
            processed = sketch.copy()
        
        # Apply denoising
        if self.denoise:
            processed = self._denoise(processed)
        
        # Enhance edges
        if self.enhance_edges:
            processed = self._enhance_edges(processed)
        
        # Threshold to binary
        processed = self._threshold(processed)
        
        # Invert if requested (for AI models that expect dark lines on white)
        if self.invert:
            processed = cv2.bitwise_not(processed)
        
        # Resize to target size
        processed = self._resize_with_padding(processed)
        
        return processed
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply denoising to remove small artifacts."""
        # Use morphological opening to remove small noise
        kernel = np.ones((2, 2), np.uint8)
        denoised = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        # Apply bilateral filter for edge-preserving smoothing
        denoised = cv2.bilateralFilter(denoised, 5, 50, 50)
        
        return denoised
    
    def _enhance_edges(self, image: np.ndarray) -> np.ndarray:
        """Enhance edges for cleaner lines."""
        # Apply slight Gaussian blur first
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Enhance using unsharp masking
        enhanced = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
        
        return enhanced
    
    def _threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding for clean binary output."""
        # Use Otsu's thresholding for automatic threshold detection
        _, binary = cv2.threshold(
            image, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        return binary
    
    def _resize_with_padding(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size while maintaining aspect ratio.
        Adds padding to fill remaining space.
        """
        h, w = image.shape[:2]
        target_w, target_h = self.target_size
        
        # Calculate scale to fit within target
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create padded output (white background if inverted, black otherwise)
        pad_value = 255 if self.invert else 0
        padded = np.full((target_h, target_w), pad_value, dtype=np.uint8)
        
        # Center the resized image
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return padded
    
    def to_pil(self, sketch: np.ndarray) -> Image.Image:
        """Convert processed sketch to PIL Image."""
        return Image.fromarray(sketch)
    
    def to_base64(self, sketch: np.ndarray, format: str = 'PNG') -> str:
        """
        Convert sketch to base64 encoded string.
        
        Args:
            sketch: Processed sketch image
            format: Image format ('PNG', 'JPEG')
            
        Returns:
            Base64 encoded string
        """
        pil_image = self.to_pil(sketch)
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def analyze_sketch(self, sketch: np.ndarray) -> dict:
        """
        Analyze sketch properties for prompt generation.
        
        Args:
            sketch: Raw or processed sketch
            
        Returns:
            Dict with analysis results
        """
        # Convert to grayscale if needed
        if len(sketch.shape) == 3:
            gray = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
        else:
            gray = sketch.copy()
        
        # Calculate stroke density
        total_pixels = gray.shape[0] * gray.shape[1]
        stroke_pixels = np.count_nonzero(gray)
        density = stroke_pixels / total_pixels
        
        # Detect edges for complexity estimation
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.count_nonzero(edges)
        complexity = edge_pixels / total_pixels
        
        # Find contours for shape analysis
        contours, _ = cv2.findContours(
            gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        num_shapes = len(contours)
        
        # Estimate if sketch is detailed or simple
        detail_level = 'detailed' if complexity > 0.05 else 'simple'
        
        return {
            'density': density,
            'complexity': complexity,
            'num_shapes': num_shapes,
            'detail_level': detail_level,
            'width': sketch.shape[1],
            'height': sketch.shape[0],
            'has_content': stroke_pixels > 100
        }
    
    def generate_prompt_hint(self, analysis: dict) -> str:
        """
        Generate a prompt hint based on sketch analysis.
        
        Args:
            analysis: Results from analyze_sketch
            
        Returns:
            Prompt hint string
        """
        hints = []
        
        if analysis['detail_level'] == 'detailed':
            hints.append('detailed')
        else:
            hints.append('simple')
        
        if analysis['density'] > 0.3:
            hints.append('filled')
        else:
            hints.append('line art')
        
        if analysis['num_shapes'] > 5:
            hints.append('complex composition')
        elif analysis['num_shapes'] == 1:
            hints.append('single subject')
        
        return ', '.join(hints)


class SketchEnhancer:
    """
    Advanced sketch enhancement using various filters and techniques.
    """
    
    @staticmethod
    def smooth_strokes(sketch: np.ndarray, strength: int = 5) -> np.ndarray:
        """
        Smooth jagged strokes for cleaner lines.
        
        Args:
            sketch: Input sketch
            strength: Smoothing strength (1-10)
            
        Returns:
            Smoothed sketch
        """
        strength = max(1, min(strength, 10))
        kernel_size = strength * 2 + 1
        
        # Apply morphological closing to connect nearby strokes
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        smoothed = cv2.morphologyEx(sketch, cv2.MORPH_CLOSE, kernel)
        
        return smoothed
    
    @staticmethod
    def thicken_lines(sketch: np.ndarray, amount: int = 2) -> np.ndarray:
        """
        Thicken sketch lines.
        
        Args:
            sketch: Input sketch
            amount: Dilation amount
            
        Returns:
            Sketch with thicker lines
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (amount, amount))
        return cv2.dilate(sketch, kernel, iterations=1)
    
    @staticmethod
    def thin_lines(sketch: np.ndarray, amount: int = 1) -> np.ndarray:
        """
        Thin sketch lines.
        
        Args:
            sketch: Input sketch
            amount: Erosion amount
            
        Returns:
            Sketch with thinner lines
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (amount, amount))
        return cv2.erode(sketch, kernel, iterations=1)
    
    @staticmethod
    def remove_small_components(
        sketch: np.ndarray,
        min_size: int = 50
    ) -> np.ndarray:
        """
        Remove small connected components (noise).
        
        Args:
            sketch: Input sketch
            min_size: Minimum component size to keep
            
        Returns:
            Cleaned sketch
        """
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            sketch, connectivity=8
        )
        
        # Create output image
        cleaned = np.zeros_like(sketch)
        
        # Keep only large components
        for i in range(1, num_labels):  # Skip background (0)
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                cleaned[labels == i] = 255
        
        return cleaned
    
    @staticmethod
    def auto_crop(
        sketch: np.ndarray,
        padding: int = 20
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Automatically crop to content with padding.
        
        Args:
            sketch: Input sketch
            padding: Padding around content
            
        Returns:
            Tuple of (cropped sketch, bounding box)
        """
        # Find non-zero pixels
        coords = cv2.findNonZero(sketch)
        
        if coords is None:
            return sketch, (0, 0, sketch.shape[1], sketch.shape[0])
        
        x, y, w, h = cv2.boundingRect(coords)
        
        # Add padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(sketch.shape[1], x + w + padding)
        y2 = min(sketch.shape[0], y + h + padding)
        
        cropped = sketch[y1:y2, x1:x2]
        
        return cropped, (x1, y1, x2 - x1, y2 - y1)


def create_canny_sketch(image: np.ndarray) -> np.ndarray:
    """
    Create a sketch-like image from a photograph using Canny edge detection.
    Useful for testing the AI generation pipeline.
    
    Args:
        image: Input BGR image
        
    Returns:
        Sketch-like grayscale image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter for edge-preserving smoothing
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Detect edges
    edges = cv2.Canny(filtered, 30, 100)
    
    # Dilate slightly for thicker lines
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    return edges


if __name__ == "__main__":
    # Test sketch processor
    print("Testing Sketch Processor Module")
    print("=" * 40)
    
    # Create a test sketch
    test_sketch = np.zeros((400, 400), dtype=np.uint8)
    
    # Draw some test shapes
    cv2.circle(test_sketch, (200, 150), 50, 255, 3)
    cv2.rectangle(test_sketch, (100, 250), (300, 350), 255, 3)
    cv2.line(test_sketch, (50, 50), (350, 50), 255, 3)
    
    # Add some noise
    noise = np.random.randint(0, 50, test_sketch.shape, dtype=np.uint8)
    noisy_sketch = cv2.add(test_sketch, noise)
    
    # Process the sketch
    processor = SketchProcessor(target_size=(512, 512))
    processed = processor.process(noisy_sketch)
    
    # Analyze
    analysis = processor.analyze_sketch(test_sketch)
    print("Sketch Analysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")
    
    print(f"\nPrompt hint: {processor.generate_prompt_hint(analysis)}")
    
    # Display results
    cv2.imshow("Original", test_sketch)
    cv2.imshow("Noisy", noisy_sketch)
    cv2.imshow("Processed", processed)
    
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
