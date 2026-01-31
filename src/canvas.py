"""
Canvas Module - Virtual Drawing Board
======================================
Provides a transparent drawing canvas that can be overlaid on video feed.
Supports strokes, erasing, undo, and canvas management.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Deque
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto
import time


class BrushType(Enum):
    """Types of brushes available for drawing."""
    ROUND = auto()
    SQUARE = auto()
    MARKER = auto()


@dataclass
class Stroke:
    """
    Represents a single stroke on the canvas.
    
    Attributes:
        points: List of (x, y) points in the stroke
        color: BGR color of the stroke
        thickness: Line thickness
        brush_type: Type of brush used
        timestamp: When the stroke was created
    """
    points: List[Tuple[int, int]] = field(default_factory=list)
    color: Tuple[int, int, int] = (255, 255, 255)
    thickness: int = 5
    brush_type: BrushType = BrushType.ROUND
    timestamp: float = field(default_factory=time.time)
    
    def add_point(self, point: Tuple[int, int]):
        """Add a point to the stroke."""
        self.points.append(point)
    
    def is_empty(self) -> bool:
        """Check if the stroke has no points."""
        return len(self.points) == 0


class Canvas:
    """
    Virtual drawing canvas with transparency support.
    
    Supports drawing strokes, erasing, undo/redo, and various
    brush settings. Can be overlaid on camera feed.
    """
    
    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        background_color: Optional[Tuple[int, int, int]] = None
    ):
        """
        Initialize the canvas.
        
        Args:
            width: Canvas width in pixels
            height: Canvas height in pixels
            background_color: Background color (None for transparent)
        """
        self.width = width
        self.height = height
        self.background_color = background_color
        
        # Main canvas (BGRA for transparency)
        self._canvas = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Stroke management
        self._strokes: List[Stroke] = []
        self._current_stroke: Optional[Stroke] = None
        self._undo_stack: Deque[List[Stroke]] = deque(maxlen=50)
        self._redo_stack: Deque[List[Stroke]] = deque(maxlen=50)
        
        # Drawing settings
        self._brush_color = (255, 255, 255)  # White
        self._brush_thickness = 5
        self._brush_type = BrushType.ROUND
        
        # Eraser settings
        self._eraser_size = 30
        self._is_erasing = False
        
        # Position tracking for smooth lines
        self._last_position: Optional[Tuple[int, int]] = None
        
        # Performance optimization
        self._needs_redraw = True
        self._cached_canvas: Optional[np.ndarray] = None
    
    def start_stroke(self, position: Tuple[int, int]):
        """
        Start a new stroke at the given position.
        
        Args:
            position: (x, y) starting position
        """
        # Save current state for undo
        self._save_undo_state()
        
        self._current_stroke = Stroke(
            points=[position],
            color=self._brush_color,
            thickness=self._brush_thickness,
            brush_type=self._brush_type
        )
        self._last_position = position
        self._needs_redraw = True
    
    def continue_stroke(self, position: Tuple[int, int]):
        """
        Continue the current stroke to a new position.
        
        Args:
            position: (x, y) new position
        """
        if self._current_stroke is None:
            self.start_stroke(position)
            return
        
        # Interpolate points for smoother lines
        if self._last_position:
            interpolated = self._interpolate_points(self._last_position, position)
            for point in interpolated:
                self._current_stroke.add_point(point)
        else:
            self._current_stroke.add_point(position)
        
        self._last_position = position
        self._needs_redraw = True
    
    def end_stroke(self):
        """End the current stroke and add it to the stroke list."""
        if self._current_stroke and not self._current_stroke.is_empty():
            self._strokes.append(self._current_stroke)
        
        self._current_stroke = None
        self._last_position = None
        self._needs_redraw = True
        self._redo_stack.clear()  # Clear redo on new stroke
    
    def erase_at(self, position: Tuple[int, int]):
        """
        Erase strokes at the given position.
        
        Args:
            position: (x, y) eraser center position
        """
        if not self._is_erasing:
            self._save_undo_state()
            self._is_erasing = True
        
        x, y = position
        
        # Remove points within eraser radius
        new_strokes = []
        for stroke in self._strokes:
            new_points = []
            for px, py in stroke.points:
                dist = np.sqrt((px - x) ** 2 + (py - y) ** 2)
                if dist > self._eraser_size:
                    new_points.append((px, py))
            
            # Keep stroke if it still has points
            if new_points:
                new_stroke = Stroke(
                    points=new_points,
                    color=stroke.color,
                    thickness=stroke.thickness,
                    brush_type=stroke.brush_type,
                    timestamp=stroke.timestamp
                )
                new_strokes.append(new_stroke)
        
        self._strokes = new_strokes
        self._needs_redraw = True
    
    def stop_erasing(self):
        """Stop erasing mode."""
        self._is_erasing = False
    
    def _interpolate_points(
        self,
        p1: Tuple[int, int],
        p2: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """
        Interpolate points between two positions for smooth lines.
        
        Args:
            p1: Start point
            p2: End point
            
        Returns:
            List of interpolated points
        """
        x1, y1 = p1
        x2, y2 = p2
        
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        if distance < 1:
            return [p2]
        
        # Number of points based on distance (ensures smooth lines)
        num_points = max(int(distance / 2), 1)
        
        points = []
        for i in range(1, num_points + 1):
            t = i / num_points
            x = int(x1 + (x2 - x1) * t)
            y = int(y1 + (y2 - y1) * t)
            points.append((x, y))
        
        return points
    
    def _save_undo_state(self):
        """Save current state for undo."""
        # Deep copy strokes
        state = []
        for stroke in self._strokes:
            new_stroke = Stroke(
                points=stroke.points.copy(),
                color=stroke.color,
                thickness=stroke.thickness,
                brush_type=stroke.brush_type,
                timestamp=stroke.timestamp
            )
            state.append(new_stroke)
        self._undo_stack.append(state)
    
    def undo(self) -> bool:
        """
        Undo the last stroke.
        
        Returns:
            True if undo was successful, False if nothing to undo
        """
        if not self._undo_stack:
            return False
        
        # Save current state for redo
        current_state = []
        for stroke in self._strokes:
            new_stroke = Stroke(
                points=stroke.points.copy(),
                color=stroke.color,
                thickness=stroke.thickness,
                brush_type=stroke.brush_type,
                timestamp=stroke.timestamp
            )
            current_state.append(new_stroke)
        self._redo_stack.append(current_state)
        
        # Restore previous state
        self._strokes = self._undo_stack.pop()
        self._needs_redraw = True
        return True
    
    def redo(self) -> bool:
        """
        Redo the last undone stroke.
        
        Returns:
            True if redo was successful, False if nothing to redo
        """
        if not self._redo_stack:
            return False
        
        # Save current state for undo
        self._save_undo_state()
        
        # Restore redo state
        self._strokes = self._redo_stack.pop()
        self._needs_redraw = True
        return True
    
    def clear(self):
        """Clear all strokes from the canvas."""
        if self._strokes:  # Only save undo if there's something to clear
            self._save_undo_state()
        
        self._strokes.clear()
        self._current_stroke = None
        self._last_position = None
        self._needs_redraw = True
    
    def _render_canvas(self) -> np.ndarray:
        """
        Render all strokes to the canvas.
        
        Returns:
            Rendered canvas as BGRA image
        """
        # Create fresh canvas
        canvas = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        
        # Draw all strokes
        all_strokes = self._strokes.copy()
        if self._current_stroke:
            all_strokes.append(self._current_stroke)
        
        for stroke in all_strokes:
            if len(stroke.points) < 2:
                # Single point - draw circle
                if stroke.points:
                    cv2.circle(
                        canvas,
                        stroke.points[0],
                        stroke.thickness // 2,
                        (*stroke.color, 255),  # Add alpha
                        -1
                    )
                continue
            
            # Draw stroke as connected lines
            for i in range(len(stroke.points) - 1):
                p1 = stroke.points[i]
                p2 = stroke.points[i + 1]
                
                if stroke.brush_type == BrushType.ROUND:
                    cv2.line(canvas, p1, p2, (*stroke.color, 255), stroke.thickness)
                    cv2.circle(canvas, p2, stroke.thickness // 2, (*stroke.color, 255), -1)
                elif stroke.brush_type == BrushType.SQUARE:
                    cv2.line(canvas, p1, p2, (*stroke.color, 255), stroke.thickness, cv2.LINE_4)
                elif stroke.brush_type == BrushType.MARKER:
                    # Semi-transparent marker effect
                    cv2.line(canvas, p1, p2, (*stroke.color, 128), stroke.thickness * 2)
        
        return canvas
    
    def get_canvas(self) -> np.ndarray:
        """
        Get the current canvas as BGRA image.
        
        Returns:
            Canvas as numpy array with alpha channel
        """
        if self._needs_redraw or self._cached_canvas is None:
            self._cached_canvas = self._render_canvas()
            self._needs_redraw = False
        return self._cached_canvas.copy()
    
    def get_canvas_bgr(self) -> np.ndarray:
        """
        Get the canvas as BGR image (no transparency).
        
        Returns:
            Canvas as BGR numpy array
        """
        canvas = self.get_canvas()
        return cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)
    
    def get_sketch_image(self) -> np.ndarray:
        """
        Get a clean sketch image for AI processing.
        White strokes on black background.
        
        Returns:
            Sketch as grayscale numpy array
        """
        # Create black background
        sketch = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Draw all strokes in white
        for stroke in self._strokes:
            if len(stroke.points) < 2:
                if stroke.points:
                    cv2.circle(sketch, stroke.points[0], stroke.thickness // 2, 255, -1)
                continue
            
            for i in range(len(stroke.points) - 1):
                p1 = stroke.points[i]
                p2 = stroke.points[i + 1]
                cv2.line(sketch, p1, p2, 255, stroke.thickness)
                cv2.circle(sketch, p2, stroke.thickness // 2, 255, -1)
        
        return sketch
    
    def overlay_on_frame(
        self,
        frame: np.ndarray,
        alpha: float = 0.8
    ) -> np.ndarray:
        """
        Overlay the canvas on a video frame.
        
        Args:
            frame: BGR video frame
            alpha: Opacity of the canvas (0-1)
            
        Returns:
            Frame with canvas overlay
        """
        # Resize canvas if needed
        if frame.shape[:2] != (self.height, self.width):
            self.resize(frame.shape[1], frame.shape[0])
        
        canvas = self.get_canvas()
        
        # Extract alpha channel
        canvas_alpha = canvas[:, :, 3] / 255.0 * alpha
        canvas_bgr = canvas[:, :, :3]
        
        # Blend
        result = frame.copy()
        for c in range(3):
            result[:, :, c] = (
                result[:, :, c] * (1 - canvas_alpha) +
                canvas_bgr[:, :, c] * canvas_alpha
            ).astype(np.uint8)
        
        return result
    
    def resize(self, new_width: int, new_height: int):
        """
        Resize the canvas and scale all strokes.
        
        Args:
            new_width: New canvas width
            new_height: New canvas height
        """
        if new_width == self.width and new_height == self.height:
            return
        
        # Scale factors
        scale_x = new_width / self.width
        scale_y = new_height / self.height
        
        # Scale all stroke points
        for stroke in self._strokes:
            stroke.points = [
                (int(x * scale_x), int(y * scale_y))
                for x, y in stroke.points
            ]
        
        self.width = new_width
        self.height = new_height
        self._needs_redraw = True
    
    # Brush settings
    def set_brush_color(self, color: Tuple[int, int, int]):
        """Set the brush color (BGR)."""
        self._brush_color = color
    
    def set_brush_thickness(self, thickness: int):
        """Set the brush thickness."""
        self._brush_thickness = max(1, min(thickness, 50))
    
    def set_brush_type(self, brush_type: BrushType):
        """Set the brush type."""
        self._brush_type = brush_type
    
    def set_eraser_size(self, size: int):
        """Set the eraser size."""
        self._eraser_size = max(10, min(size, 100))
    
    def get_brush_settings(self) -> dict:
        """Get current brush settings."""
        return {
            'color': self._brush_color,
            'thickness': self._brush_thickness,
            'type': self._brush_type,
            'eraser_size': self._eraser_size
        }
    
    def has_content(self) -> bool:
        """Check if the canvas has any strokes."""
        return len(self._strokes) > 0
    
    def get_stroke_count(self) -> int:
        """Get the number of strokes."""
        return len(self._strokes)
    
    def get_point_count(self) -> int:
        """Get total number of points across all strokes."""
        return sum(len(s.points) for s in self._strokes)


class ColorPalette:
    """Predefined color palette for drawing."""
    
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    YELLOW = (0, 255, 255)
    CYAN = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    ORANGE = (0, 165, 255)
    PURPLE = (128, 0, 128)
    
    @classmethod
    def get_all(cls) -> List[Tuple[int, int, int]]:
        """Get all palette colors."""
        return [
            cls.WHITE, cls.RED, cls.GREEN, cls.BLUE,
            cls.YELLOW, cls.CYAN, cls.MAGENTA, cls.ORANGE, cls.PURPLE
        ]


if __name__ == "__main__":
    # Test canvas module
    print("Testing Canvas Module")
    print("=" * 40)
    print("Controls:")
    print("  - Left mouse: Draw")
    print("  - Right mouse: Erase")
    print("  - 'c': Clear canvas")
    print("  - 'u': Undo")
    print("  - 'r': Redo")
    print("  - 'q': Quit")
    
    canvas = Canvas(800, 600)
    canvas.set_brush_color(ColorPalette.WHITE)
    canvas.set_brush_thickness(5)
    
    drawing = False
    erasing = False
    
    def mouse_callback(event, x, y, flags, param):
        global drawing, erasing
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            canvas.start_stroke((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            canvas.end_stroke()
        elif event == cv2.EVENT_RBUTTONDOWN:
            erasing = True
        elif event == cv2.EVENT_RBUTTONUP:
            erasing = False
            canvas.stop_erasing()
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                canvas.continue_stroke((x, y))
            elif erasing:
                canvas.erase_at((x, y))
    
    cv2.namedWindow("Canvas Test")
    cv2.setMouseCallback("Canvas Test", mouse_callback)
    
    while True:
        # Create display image
        display = np.zeros((600, 800, 3), dtype=np.uint8)
        display[:] = (50, 50, 50)  # Dark gray background
        
        # Overlay canvas
        display = canvas.overlay_on_frame(display)
        
        # Show info
        cv2.putText(
            display, f"Strokes: {canvas.get_stroke_count()}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        
        cv2.imshow("Canvas Test", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas.clear()
        elif key == ord('u'):
            canvas.undo()
        elif key == ord('r'):
            canvas.redo()
    
    cv2.destroyAllWindows()
