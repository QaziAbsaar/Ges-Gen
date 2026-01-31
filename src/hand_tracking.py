"""
Hand Tracking Module - MediaPipe Hand Landmark Detection
=========================================================
Detects and tracks hand landmarks using MediaPipe.
Provides normalized and pixel coordinates for all 21 hand landmarks.

Updated for MediaPipe 0.10.30+ (Tasks API) with VIDEO mode for performance.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from enum import IntEnum
import urllib.request
from pathlib import Path
import time


class HandLandmark(IntEnum):
    """
    MediaPipe hand landmark indices.
    Reference: https://mediapipe.dev/images/mobile/hand_landmarks.png
    """
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


@dataclass
class Point:
    """Represents a 2D/3D point with normalized and pixel coordinates."""
    x: float  # Normalized x (0-1)
    y: float  # Normalized y (0-1)
    z: float  # Normalized z (depth)
    px: int   # Pixel x coordinate
    py: int   # Pixel y coordinate
    
    def to_tuple(self) -> Tuple[int, int]:
        """Return pixel coordinates as tuple."""
        return (self.px, self.py)
    
    def distance_to(self, other: 'Point') -> float:
        """Calculate Euclidean distance to another point (normalized)."""
        return np.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )
    
    def pixel_distance_to(self, other: 'Point') -> float:
        """Calculate Euclidean distance in pixel coordinates."""
        return np.sqrt(
            (self.px - other.px) ** 2 +
            (self.py - other.py) ** 2
        )


@dataclass
class HandData:
    """
    Contains all data for a detected hand.
    
    Attributes:
        landmarks: Dict mapping HandLandmark to Point
        handedness: 'Left' or 'Right'
        confidence: Detection confidence score
        bbox: Bounding box (x, y, w, h) in pixels
    """
    landmarks: Dict[HandLandmark, Point]
    handedness: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    
    def get_landmark(self, landmark: HandLandmark) -> Optional[Point]:
        """Get a specific landmark point."""
        return self.landmarks.get(landmark)
    
    def get_fingertip(self, finger: str) -> Optional[Point]:
        """
        Get fingertip point by finger name.
        
        Args:
            finger: One of 'thumb', 'index', 'middle', 'ring', 'pinky'
        """
        finger_map = {
            'thumb': HandLandmark.THUMB_TIP,
            'index': HandLandmark.INDEX_TIP,
            'middle': HandLandmark.MIDDLE_TIP,
            'ring': HandLandmark.RING_TIP,
            'pinky': HandLandmark.PINKY_TIP
        }
        landmark = finger_map.get(finger.lower())
        return self.landmarks.get(landmark) if landmark else None


def _download_model(model_path: Path) -> None:
    """Download the hand landmarker model if not present."""
    model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    
    print(f"[INFO] Downloading hand landmarker model...")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(model_url, model_path)
    print(f"[INFO] Model downloaded to {model_path}")


class HandTracker:
    """
    Hand tracking using MediaPipe Hand Landmarker (Tasks API).
    
    Uses VIDEO running mode for optimal performance with sequential frames.
    This mode uses tracking between frames, reducing detection overhead.
    """
    
    def __init__(
        self,
        max_hands: int = 1,
        min_detection_confidence: float = 0.5,  # Lower for speed
        min_tracking_confidence: float = 0.5,   # Lower for speed
        model_complexity: int = 0  # Kept for API compatibility
    ):
        """
        Initialize the hand tracker.
        
        Args:
            max_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            model_complexity: Kept for backward compatibility (not used)
        """
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Model path - download if not exists
        self._model_dir = Path(__file__).parent.parent / "models"
        self._model_path = self._model_dir / "hand_landmarker.task"
        
        if not self._model_path.exists():
            _download_model(self._model_path)
        
        # Create the hand landmarker using Tasks API with VIDEO mode
        # VIDEO mode is optimized for sequential frames and uses tracking
        base_options = python.BaseOptions(model_asset_path=str(self._model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,  # VIDEO mode for better FPS
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            min_hand_presence_confidence=min_detection_confidence
        )
        
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Timestamp tracking for VIDEO mode
        self._frame_timestamp_ms = 0
        self._start_time = time.time()
        
        # Store frame dimensions for coordinate conversion
        self._frame_width = 0
        self._frame_height = 0
        
        # Smoothing buffer for landmark positions
        self._smoothing_buffer: Dict[HandLandmark, List[Point]] = {}
        self._smoothing_window = 2  # Reduced for lower latency
    
    def process(self, frame: np.ndarray, smooth: bool = True) -> List[HandData]:
        """
        Process a frame and detect hands.
        
        Args:
            frame: BGR image from camera
            smooth: Whether to apply smoothing to landmark positions
            
        Returns:
            List of HandData objects for each detected hand
        """
        self._frame_height, self._frame_width = frame.shape[:2]
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Calculate timestamp in milliseconds for VIDEO mode
        # Timestamps must be monotonically increasing
        self._frame_timestamp_ms = int((time.time() - self._start_time) * 1000)
        
        # Detect hands using Tasks API in VIDEO mode
        results = self.detector.detect_for_video(mp_image, self._frame_timestamp_ms)
        
        hands_data = []
        
        if results.hand_landmarks:
            for idx, hand_landmarks in enumerate(results.hand_landmarks):
                # Get handedness
                handedness = "Right"
                confidence = 0.0
                if results.handedness and idx < len(results.handedness):
                    hand_info = results.handedness[idx]
                    if hand_info:
                        handedness = hand_info[0].category_name
                        confidence = hand_info[0].score
                
                # Convert landmarks to our Point format
                landmarks = {}
                min_x, min_y = float('inf'), float('inf')
                max_x, max_y = 0, 0
                
                for lm_idx, lm in enumerate(hand_landmarks):
                    # Convert normalized coordinates to pixel coordinates
                    px = int(lm.x * self._frame_width)
                    py = int(lm.y * self._frame_height)
                    
                    # Clamp to frame bounds
                    px = max(0, min(px, self._frame_width - 1))
                    py = max(0, min(py, self._frame_height - 1))
                    
                    point = Point(
                        x=lm.x,
                        y=lm.y,
                        z=lm.z if hasattr(lm, 'z') else 0.0,
                        px=px,
                        py=py
                    )
                    
                    # Apply smoothing if enabled
                    if smooth:
                        point = self._smooth_point(HandLandmark(lm_idx), point)
                    
                    landmarks[HandLandmark(lm_idx)] = point
                    
                    # Update bounding box
                    min_x = min(min_x, px)
                    min_y = min(min_y, py)
                    max_x = max(max_x, px)
                    max_y = max(max_y, py)
                
                # Calculate bounding box with padding
                padding = 20
                bbox = (
                    max(0, min_x - padding),
                    max(0, min_y - padding),
                    min(self._frame_width, max_x - min_x + 2 * padding),
                    min(self._frame_height, max_y - min_y + 2 * padding)
                )
                
                hands_data.append(HandData(
                    landmarks=landmarks,
                    handedness=handedness,
                    confidence=confidence,
                    bbox=bbox
                ))
        
        return hands_data
    
    def _smooth_point(self, landmark: HandLandmark, point: Point) -> Point:
        """
        Apply temporal smoothing to a landmark point.
        Uses a simple moving average filter.
        
        Args:
            landmark: The landmark identifier
            point: Current point position
            
        Returns:
            Smoothed point
        """
        if landmark not in self._smoothing_buffer:
            self._smoothing_buffer[landmark] = []
        
        buffer = self._smoothing_buffer[landmark]
        buffer.append(point)
        
        # Keep only recent points
        if len(buffer) > self._smoothing_window:
            buffer.pop(0)
        
        # Calculate average
        if len(buffer) == 1:
            return point
        
        avg_x = sum(p.x for p in buffer) / len(buffer)
        avg_y = sum(p.y for p in buffer) / len(buffer)
        avg_z = sum(p.z for p in buffer) / len(buffer)
        avg_px = int(sum(p.px for p in buffer) / len(buffer))
        avg_py = int(sum(p.py for p in buffer) / len(buffer))
        
        return Point(x=avg_x, y=avg_y, z=avg_z, px=avg_px, py=avg_py)
    
    def draw_landmarks(
        self,
        frame: np.ndarray,
        hand_data: HandData,
        draw_connections: bool = True,
        landmark_color: Tuple[int, int, int] = (0, 255, 0),
        connection_color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw hand landmarks on frame.
        
        Args:
            frame: Image to draw on
            hand_data: Hand data to visualize
            draw_connections: Whether to draw connections between landmarks
            landmark_color: BGR color for landmarks
            connection_color: BGR color for connections
            thickness: Line thickness
            
        Returns:
            Frame with landmarks drawn
        """
        # Define finger connections
        connections = [
            # Thumb
            (HandLandmark.WRIST, HandLandmark.THUMB_CMC),
            (HandLandmark.THUMB_CMC, HandLandmark.THUMB_MCP),
            (HandLandmark.THUMB_MCP, HandLandmark.THUMB_IP),
            (HandLandmark.THUMB_IP, HandLandmark.THUMB_TIP),
            # Index
            (HandLandmark.WRIST, HandLandmark.INDEX_MCP),
            (HandLandmark.INDEX_MCP, HandLandmark.INDEX_PIP),
            (HandLandmark.INDEX_PIP, HandLandmark.INDEX_DIP),
            (HandLandmark.INDEX_DIP, HandLandmark.INDEX_TIP),
            # Middle
            (HandLandmark.WRIST, HandLandmark.MIDDLE_MCP),
            (HandLandmark.MIDDLE_MCP, HandLandmark.MIDDLE_PIP),
            (HandLandmark.MIDDLE_PIP, HandLandmark.MIDDLE_DIP),
            (HandLandmark.MIDDLE_DIP, HandLandmark.MIDDLE_TIP),
            # Ring
            (HandLandmark.WRIST, HandLandmark.RING_MCP),
            (HandLandmark.RING_MCP, HandLandmark.RING_PIP),
            (HandLandmark.RING_PIP, HandLandmark.RING_DIP),
            (HandLandmark.RING_DIP, HandLandmark.RING_TIP),
            # Pinky
            (HandLandmark.WRIST, HandLandmark.PINKY_MCP),
            (HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP),
            (HandLandmark.PINKY_PIP, HandLandmark.PINKY_DIP),
            (HandLandmark.PINKY_DIP, HandLandmark.PINKY_TIP),
            # Palm
            (HandLandmark.INDEX_MCP, HandLandmark.MIDDLE_MCP),
            (HandLandmark.MIDDLE_MCP, HandLandmark.RING_MCP),
            (HandLandmark.RING_MCP, HandLandmark.PINKY_MCP),
        ]
        
        # Draw connections
        if draw_connections:
            for start, end in connections:
                p1 = hand_data.landmarks.get(start)
                p2 = hand_data.landmarks.get(end)
                if p1 and p2:
                    cv2.line(frame, p1.to_tuple(), p2.to_tuple(),
                            connection_color, thickness)
        
        # Draw landmarks
        for landmark, point in hand_data.landmarks.items():
            # Different colors for fingertips
            if landmark in [HandLandmark.THUMB_TIP, HandLandmark.INDEX_TIP,
                           HandLandmark.MIDDLE_TIP, HandLandmark.RING_TIP,
                           HandLandmark.PINKY_TIP]:
                color = (0, 0, 255)  # Red for fingertips
                radius = 8
            else:
                color = landmark_color
                radius = 5
            
            cv2.circle(frame, point.to_tuple(), radius, color, -1)
            cv2.circle(frame, point.to_tuple(), radius, (0, 0, 0), 1)
        
        return frame
    
    def reset_smoothing(self):
        """Reset the smoothing buffer."""
        self._smoothing_buffer.clear()
    
    def release(self):
        """Release resources."""
        if self.detector:
            self.detector.close()


def calculate_finger_states(hand_data: HandData) -> Dict[str, bool]:
    """
    Determine which fingers are extended.
    
    Uses the relative positions of fingertips to their corresponding
    MCP (knuckle) joints to determine if a finger is extended.
    
    Args:
        hand_data: Hand data with landmarks
        
    Returns:
        Dict with finger names as keys and extension state as values
    """
    landmarks = hand_data.landmarks
    
    # Finger tip and base pairs
    fingers = {
        'thumb': (HandLandmark.THUMB_TIP, HandLandmark.THUMB_MCP, HandLandmark.THUMB_IP),
        'index': (HandLandmark.INDEX_TIP, HandLandmark.INDEX_MCP, HandLandmark.INDEX_PIP),
        'middle': (HandLandmark.MIDDLE_TIP, HandLandmark.MIDDLE_MCP, HandLandmark.MIDDLE_PIP),
        'ring': (HandLandmark.RING_TIP, HandLandmark.RING_MCP, HandLandmark.RING_PIP),
        'pinky': (HandLandmark.PINKY_TIP, HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP)
    }
    
    states = {}
    wrist = landmarks.get(HandLandmark.WRIST)
    
    for finger, (tip_lm, mcp_lm, pip_lm) in fingers.items():
        tip = landmarks.get(tip_lm)
        mcp = landmarks.get(mcp_lm)
        pip = landmarks.get(pip_lm)
        
        if not all([tip, mcp, pip]):
            states[finger] = False
            continue
        
        if finger == 'thumb':
            # Thumb: check if tip is to the side of the IP joint
            # Account for left/right hand
            if hand_data.handedness == 'Right':
                states[finger] = tip.x < pip.x  # Thumb extended outward
            else:
                states[finger] = tip.x > pip.x
        else:
            # Other fingers: tip should be above PIP (lower y value)
            # Also check that tip is extended past PIP
            states[finger] = tip.y < pip.y
    
    return states


if __name__ == "__main__":
    # Test hand tracking module
    from camera import Camera
    
    print("Testing Hand Tracking Module")
    print("=" * 40)
    print("Press 'q' to quit")
    
    with Camera() as cam:
        tracker = HandTracker(max_hands=1)
        
        while True:
            frame = cam.get_frame()
            if frame is None:
                continue
            
            # Process frame
            hands = tracker.process(frame)
            
            # Draw results
            for hand in hands:
                frame = tracker.draw_landmarks(frame, hand)
                
                # Show finger states
                states = calculate_finger_states(hand)
                y_pos = 30
                for finger, extended in states.items():
                    status = "UP" if extended else "DOWN"
                    color = (0, 255, 0) if extended else (0, 0, 255)
                    cv2.putText(frame, f"{finger}: {status}", (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_pos += 25
            
            # Show FPS
            fps = cam.get_fps()
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Hand Tracking Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        tracker.release()
        cv2.destroyAllWindows()
