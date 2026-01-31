"""
UI Module - Main Application Interface
======================================
Real-time gesture drawing interface with AI image generation.
Combines all modules into a cohesive application.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import time
import os
from pathlib import Path
from datetime import datetime

# Import project modules
from camera import Camera
from hand_tracking import HandTracker, HandData
from gesture_logic import GestureDetector, Gesture, GestureState, draw_gesture_ui
from canvas import Canvas, ColorPalette, BrushType
from sketch_processor import SketchProcessor
from image_generator import (
    create_generator, GenerationRequest, GenerationResult,
    StylePresets, ImageGenerator
)


class GestureDrawingApp:
    """
    Main application class for gesture-based drawing with AI generation.
    
    Combines webcam capture, hand tracking, gesture recognition,
    canvas drawing, and AI image generation into a unified interface.
    """
    
    # Window dimensions - reduced for better performance
    MAIN_WIDTH = 960   # Reduced from 1280
    MAIN_HEIGHT = 540  # Reduced from 720
    
    # UI Colors (BGR)
    UI_BG_COLOR = (30, 30, 30)
    UI_ACCENT_COLOR = (0, 200, 255)
    UI_TEXT_COLOR = (255, 255, 255)
    UI_SUCCESS_COLOR = (0, 255, 0)
    UI_ERROR_COLOR = (0, 0, 255)
    
    def __init__(self, camera_id: int = 0, use_mock_generator: bool = False):
        """
        Initialize the application.
        
        Args:
            camera_id: Camera device index
            use_mock_generator: Use mock generator for testing
        """
        # Initialize components
        self.camera = Camera(
            camera_id=camera_id,
            width=self.MAIN_WIDTH,
            height=self.MAIN_HEIGHT,
            fps=30
        )
        
        # Lower confidence thresholds for better performance
        self.hand_tracker = HandTracker(
            max_hands=1,
            min_detection_confidence=0.5,  # Lower for speed
            min_tracking_confidence=0.5,   # Lower for speed
            model_complexity=0
        )
        
        self.gesture_detector = GestureDetector(smoothing_window=3)  # Reduced
        
        self.canvas = Canvas(
            width=self.MAIN_WIDTH,
            height=self.MAIN_HEIGHT
        )
        self.canvas.set_brush_color(ColorPalette.WHITE)
        self.canvas.set_brush_thickness(8)
        
        self.sketch_processor = SketchProcessor(
            target_size=(512, 512),
            invert=True
        )
        
        self.generator = create_generator(use_mock=use_mock_generator)
        
        # Application state
        self._running = False
        self._is_drawing = False
        self._was_drawing = False
        
        # Generated image display
        self._generated_image: Optional[np.ndarray] = None
        self._show_generated = False
        self._generation_status = ""
        
        # Style selection
        self._styles = StylePresets.get_all_names()
        self._current_style_idx = 0
        
        # Color palette
        self._colors = ColorPalette.get_all()
        self._current_color_idx = 0
        
        # Brush sizes
        self._brush_sizes = [3, 5, 8, 12, 18, 25]
        self._current_size_idx = 2
        
        # Save directory
        self._save_dir = Path("output")
        self._save_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self._fps_counter = 0
        self._fps_time = time.time()
        self._current_fps = 0.0
        
        # Register gesture callbacks
        self.gesture_detector.register_callback(Gesture.SUBMIT, self._on_submit)
        self.gesture_detector.register_callback(Gesture.CLEAR, self._on_clear)
    
    def _on_submit(self):
        """Callback when submit gesture is triggered."""
        if not self.canvas.has_content():
            self._generation_status = "Canvas is empty!"
            return
        
        if self.generator.is_generating():
            self._generation_status = "Generation in progress..."
            return
        
        self._generation_status = "Generating image..."
        self._submit_for_generation()
    
    def _on_clear(self):
        """Callback when clear gesture is triggered."""
        self.canvas.clear()
        self._generated_image = None
        self._show_generated = False
        self._generation_status = "Canvas cleared"
    
    def _submit_for_generation(self):
        """Submit the current sketch for AI generation."""
        # Get sketch from canvas
        sketch = self.canvas.get_sketch_image()
        
        # Process sketch
        processed = self.sketch_processor.process(sketch)
        
        # Analyze sketch for prompt hints
        analysis = self.sketch_processor.analyze_sketch(sketch)
        prompt_hint = self.sketch_processor.generate_prompt_hint(analysis)
        
        # Create generation request
        current_style = self._styles[self._current_style_idx]
        
        request = GenerationRequest(
            sketch=processed,
            prompt=f"A {prompt_hint} illustration, artistic",
            style=current_style
        )
        
        # Set completion callback
        self.generator.set_on_complete(self._on_generation_complete)
        
        # Start generation
        self.generator.generate(request, async_mode=True)
    
    def _on_generation_complete(self, result: GenerationResult):
        """Callback when AI generation completes."""
        if result.success:
            self._generated_image = result.image
            self._show_generated = True
            self._generation_status = f"Generated in {result.generation_time:.1f}s"
            
            # Auto-save
            self._save_result(result.image)
        else:
            self._generation_status = f"Error: {result.error}"
    
    def _save_result(self, image: np.ndarray):
        """Save generated image to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self._save_dir / f"generated_{timestamp}.png"
        cv2.imwrite(str(filename), image)
        print(f"[INFO] Saved: {filename}")
    
    def _save_sketch(self):
        """Save current sketch to disk."""
        sketch = self.canvas.get_sketch_image()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self._save_dir / f"sketch_{timestamp}.png"
        cv2.imwrite(str(filename), sketch)
        print(f"[INFO] Saved sketch: {filename}")
    
    def _update_fps(self):
        """Update FPS counter."""
        self._fps_counter += 1
        current_time = time.time()
        elapsed = current_time - self._fps_time
        
        if elapsed >= 1.0:
            self._current_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_time = current_time
    
    def _process_gesture(self, gesture_state: GestureState):
        """Process detected gesture and update canvas."""
        gesture = gesture_state.gesture
        position = gesture_state.position
        
        if gesture == Gesture.DRAW and position:
            # Drawing mode
            if not self._is_drawing:
                self.canvas.start_stroke(position)
                self._is_drawing = True
            else:
                self.canvas.continue_stroke(position)
        
        elif gesture == Gesture.ERASE and position:
            # Erasing mode
            if self._is_drawing:
                self.canvas.end_stroke()
                self._is_drawing = False
            self.canvas.erase_at(position)
        
        elif gesture == Gesture.UNDO:
            # Undo last stroke
            if gesture_state.is_stable and not self._was_drawing:
                if self.canvas.undo():
                    self._generation_status = "Undo"
        
        else:
            # End current stroke for any other gesture
            if self._is_drawing:
                self.canvas.end_stroke()
                self._is_drawing = False
            
            if gesture == Gesture.ERASE:
                self.canvas.stop_erasing()
        
        self._was_drawing = self._is_drawing
    
    def _draw_ui(self, frame: np.ndarray, gesture_state: GestureState) -> np.ndarray:
        """Draw UI elements on the frame."""
        h, w = frame.shape[:2]
        
        # Draw gesture UI (bottom left)
        frame = draw_gesture_ui(frame, gesture_state, self.gesture_detector)
        
        # Draw top bar
        cv2.rectangle(frame, (0, 0), (w, 50), self.UI_BG_COLOR, -1)
        
        # FPS
        cv2.putText(
            frame, f"FPS: {self._current_fps:.1f}",
            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            self.UI_SUCCESS_COLOR, 2
        )
        
        # Style selector
        style_text = f"Style: {self._styles[self._current_style_idx]}"
        cv2.putText(
            frame, style_text,
            (150, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            self.UI_ACCENT_COLOR, 2
        )
        
        # Brush size
        size_text = f"Brush: {self._brush_sizes[self._current_size_idx]}px"
        cv2.putText(
            frame, size_text,
            (400, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            self.UI_TEXT_COLOR, 2
        )
        
        # Generation status
        if self._generation_status:
            status_color = self.UI_ACCENT_COLOR
            if "Error" in self._generation_status:
                status_color = self.UI_ERROR_COLOR
            elif "Generated" in self._generation_status or "Saved" in self._generation_status:
                status_color = self.UI_SUCCESS_COLOR
            
            cv2.putText(
                frame, self._generation_status,
                (w - 350, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                status_color, 2
            )
        
        # Color palette (top right)
        palette_start = w - 300
        for i, color in enumerate(self._colors[:8]):
            x = palette_start + i * 35
            cv2.rectangle(frame, (x, 5), (x + 30, 45), color, -1)
            if i == self._current_color_idx:
                cv2.rectangle(frame, (x - 2, 3), (x + 32, 47), (255, 255, 255), 2)
        
        # Instructions (bottom right)
        instructions = [
            "[1-8] Color | [+/-] Brush",
            "[S] Style | [Space] Save",
            "[C] Clear | [Q] Quit"
        ]
        
        y_pos = h - 80
        for inst in instructions:
            cv2.putText(
                frame, inst,
                (w - 250, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (150, 150, 150), 1
            )
            y_pos += 20
        
        # Draw cursor at finger position
        if gesture_state.position:
            cursor_color = self.UI_ACCENT_COLOR
            cursor_radius = self._brush_sizes[self._current_size_idx] // 2 + 5
            
            if gesture_state.gesture == Gesture.DRAW:
                cursor_color = self._colors[self._current_color_idx]
            elif gesture_state.gesture == Gesture.ERASE:
                cursor_color = (100, 100, 100)
                cursor_radius = self.canvas._eraser_size
            
            cv2.circle(frame, gesture_state.position, cursor_radius, cursor_color, 2)
        
        return frame
    
    def _draw_generated_panel(self, frame: np.ndarray) -> np.ndarray:
        """Draw the generated image panel on the right side."""
        if not self._show_generated or self._generated_image is None:
            return frame
        
        h, w = frame.shape[:2]
        panel_width = 300
        
        # Resize generated image to fit panel
        gen_img = self._generated_image.copy()
        gen_h, gen_w = gen_img.shape[:2]
        scale = min(panel_width / gen_w, (h - 100) / gen_h)
        new_w = int(gen_w * scale)
        new_h = int(gen_h * scale)
        gen_img = cv2.resize(gen_img, (new_w, new_h))
        
        # Create panel background
        panel_x = w - panel_width - 20
        panel_y = 60
        
        cv2.rectangle(
            frame,
            (panel_x - 10, panel_y - 10),
            (panel_x + panel_width + 10, panel_y + new_h + 40),
            self.UI_BG_COLOR, -1
        )
        cv2.rectangle(
            frame,
            (panel_x - 10, panel_y - 10),
            (panel_x + panel_width + 10, panel_y + new_h + 40),
            self.UI_ACCENT_COLOR, 2
        )
        
        # Title
        cv2.putText(
            frame, "Generated Image",
            (panel_x, panel_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            self.UI_TEXT_COLOR, 2
        )
        
        # Place generated image
        img_x = panel_x + (panel_width - new_w) // 2
        img_y = panel_y + 30
        
        frame[img_y:img_y + new_h, img_x:img_x + new_w] = gen_img
        
        return frame
    
    def _handle_keyboard(self, key: int) -> bool:
        """
        Handle keyboard input.
        
        Returns:
            False if should quit, True otherwise
        """
        if key == ord('q') or key == 27:  # Q or Escape
            return False
        
        elif key == ord('c'):
            self.canvas.clear()
            self._generated_image = None
            self._show_generated = False
            self._generation_status = "Canvas cleared"
        
        elif key == ord('u'):
            if self.canvas.undo():
                self._generation_status = "Undo"
        
        elif key == ord('r'):
            if self.canvas.redo():
                self._generation_status = "Redo"
        
        elif key == ord('s'):
            # Cycle styles
            self._current_style_idx = (self._current_style_idx + 1) % len(self._styles)
            self._generation_status = f"Style: {self._styles[self._current_style_idx]}"
        
        elif key == ord(' '):  # Space - save sketch
            self._save_sketch()
            self._generation_status = "Sketch saved"
        
        elif key == ord('g'):  # G - generate
            self._on_submit()
        
        elif key == ord('+') or key == ord('='):
            self._current_size_idx = min(self._current_size_idx + 1, len(self._brush_sizes) - 1)
            self.canvas.set_brush_thickness(self._brush_sizes[self._current_size_idx])
        
        elif key == ord('-'):
            self._current_size_idx = max(self._current_size_idx - 1, 0)
            self.canvas.set_brush_thickness(self._brush_sizes[self._current_size_idx])
        
        elif ord('1') <= key <= ord('8'):
            idx = key - ord('1')
            if idx < len(self._colors):
                self._current_color_idx = idx
                self.canvas.set_brush_color(self._colors[idx])
        
        elif key == ord('h'):  # Toggle generated image
            self._show_generated = not self._show_generated
        
        return True
    
    def run(self):
        """Run the main application loop."""
        print("\n" + "=" * 60)
        print("  Gesture+Gen - Gesture-Based Drawing with AI Generation")
        print("=" * 60)
        print("\nGestures:")
        print("  ☝️  Index finger up     → Draw")
        print("  ✌️  Index + Middle up   → Pause drawing")
        print("  👌 Thumb + Index pinch → Submit to AI (hold)")
        print("  ✊ Closed fist          → Clear canvas (hold)")
        print("  🤙 Pinky only           → Undo")
        print("\nKeyboard:")
        print("  [1-8] Select color | [+/-] Brush size")
        print("  [S] Cycle style    | [G] Generate")
        print("  [U] Undo | [R] Redo | [C] Clear")
        print("  [Space] Save sketch | [H] Toggle result")
        print("  [Q] Quit")
        print("\n" + "=" * 60)
        
        # Start camera
        if not self.camera.start():
            print("[ERROR] Failed to start camera!")
            return
        
        self._running = True
        cv2.namedWindow("Gesture+Gen", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Gesture+Gen", self.MAIN_WIDTH, self.MAIN_HEIGHT)
        
        try:
            while self._running:
                # Get camera frame
                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue
                
                # Update canvas size if needed
                if frame.shape[:2] != (self.canvas.height, self.canvas.width):
                    self.canvas.resize(frame.shape[1], frame.shape[0])
                
                # Detect hands
                hands = self.hand_tracker.process(frame)
                hand_data = hands[0] if hands else None
                
                # Detect gesture
                gesture_state = self.gesture_detector.detect(hand_data)
                
                # Process gesture for drawing
                self._process_gesture(gesture_state)
                
                # Overlay canvas on frame
                display = self.canvas.overlay_on_frame(frame, alpha=0.85)
                
                # Draw hand landmarks
                if hand_data:
                    display = self.hand_tracker.draw_landmarks(display, hand_data)
                
                # Draw UI
                display = self._draw_ui(display, gesture_state)
                
                # Draw generated image panel
                display = self._draw_generated_panel(display)
                
                # Update FPS
                self._update_fps()
                
                # Show frame
                cv2.imshow("Gesture+Gen", display)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_keyboard(key):
                    break
        
        finally:
            # Cleanup
            self._running = False
            self.camera.stop()
            self.hand_tracker.release()
            cv2.destroyAllWindows()
            print("\n[INFO] Application closed")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gesture+Gen - Draw with gestures, generate with AI")
    parser.add_argument('--camera', type=int, default=0, help='Camera device index')
    parser.add_argument('--mock', action='store_true', help='Use mock generator (no API needed)')
    
    args = parser.parse_args()
    
    # Check for API key (support both HF_TOKEN and HF_API_KEY)
    api_key = os.environ.get('HF_TOKEN') or os.environ.get('HF_API_KEY')
    if not args.mock and not api_key:
        print("\n[NOTE] HF_TOKEN not set. Using mock generator.")
        print("[NOTE] For real AI generation, set HF_TOKEN in your .env file")
        print("[NOTE] Get your token at: https://huggingface.co/settings/tokens")
        args.mock = True
    
    # Create and run application
    app = GestureDrawingApp(
        camera_id=args.camera,
        use_mock_generator=args.mock
    )
    app.run()


if __name__ == "__main__":
    main()
