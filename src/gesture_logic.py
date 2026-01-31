"""
Gesture Logic Module - Gesture to Action Mapping
=================================================
Detects and interprets hand gestures for drawing control.
Implements gesture recognition using landmark positions and finger states.
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from enum import Enum, auto
from collections import deque
import time

from hand_tracking import HandData, HandLandmark, Point, calculate_finger_states


class Gesture(Enum):
    """Recognized gestures for the drawing application."""
    NONE = auto()           # No recognized gesture
    DRAW = auto()           # Index finger up - draw mode
    PAUSE = auto()          # Index + middle up - pause drawing
    ERASE = auto()          # Three fingers up - erase mode
    CLEAR = auto()          # Closed fist - clear canvas
    SUBMIT = auto()         # Thumb + index pinch - submit for AI
    UNDO = auto()           # Pinky up only - undo last stroke


@dataclass
class GestureState:
    """
    Represents the current gesture state with metadata.
    
    Attributes:
        gesture: The detected gesture
        confidence: Confidence level (0-1)
        position: Drawing position (index fingertip)
        duration: How long the gesture has been held
        is_stable: Whether the gesture is stable (held long enough)
    """
    gesture: Gesture
    confidence: float
    position: Optional[Tuple[int, int]]
    duration: float
    is_stable: bool


class GestureDetector:
    """
    Detects and tracks gestures from hand landmark data.
    
    Uses finger states and landmark distances to recognize gestures.
    Implements temporal smoothing to prevent jittery gesture changes.
    """
    
    # Distance thresholds (in normalized coordinates)
    PINCH_THRESHOLD = 0.05  # Distance for pinch detection
    FIST_THRESHOLD = 0.1    # Distance for closed fist detection
    
    # Timing thresholds
    STABILITY_TIME = 0.15   # Seconds to hold gesture for stability
    SUBMIT_HOLD_TIME = 0.5  # Seconds to hold submit gesture
    CLEAR_HOLD_TIME = 0.8   # Seconds to hold clear gesture
    
    def __init__(self, smoothing_window: int = 5):
        """
        Initialize the gesture detector.
        
        Args:
            smoothing_window: Number of frames for gesture smoothing
        """
        self.smoothing_window = smoothing_window
        
        # Gesture history for smoothing
        self._gesture_history: deque = deque(maxlen=smoothing_window)
        
        # Tracking gesture duration
        self._current_gesture = Gesture.NONE
        self._gesture_start_time = 0.0
        
        # Position smoothing
        self._position_history: deque = deque(maxlen=5)
        
        # Gesture callbacks (optional)
        self._callbacks: Dict[Gesture, callable] = {}
        
        # Special gesture states
        self._submit_triggered = False
        self._clear_triggered = False
    
    def detect(self, hand_data: Optional[HandData]) -> GestureState:
        """
        Detect the current gesture from hand data.
        
        Args:
            hand_data: Hand landmark data from tracker
            
        Returns:
            GestureState with detected gesture and metadata
        """
        if hand_data is None:
            return self._create_state(Gesture.NONE, 0.0, None)
        
        # Get finger states
        finger_states = calculate_finger_states(hand_data)
        
        # Get key landmarks
        thumb_tip = hand_data.get_fingertip('thumb')
        index_tip = hand_data.get_fingertip('index')
        middle_tip = hand_data.get_fingertip('middle')
        ring_tip = hand_data.get_fingertip('ring')
        pinky_tip = hand_data.get_fingertip('pinky')
        
        # Get position for drawing
        draw_position = index_tip.to_tuple() if index_tip else None
        
        # Detect gesture based on finger states and positions
        gesture, confidence = self._classify_gesture(
            finger_states, hand_data.landmarks,
            thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip
        )
        
        # Apply temporal smoothing
        smoothed_gesture = self._smooth_gesture(gesture)
        
        # Update position smoothing
        if draw_position:
            self._position_history.append(draw_position)
            draw_position = self._smooth_position()
        
        # Track gesture duration
        current_time = time.time()
        if smoothed_gesture != self._current_gesture:
            self._current_gesture = smoothed_gesture
            self._gesture_start_time = current_time
            self._submit_triggered = False
            self._clear_triggered = False
        
        duration = current_time - self._gesture_start_time
        is_stable = duration >= self.STABILITY_TIME
        
        # Create state
        state = self._create_state(
            smoothed_gesture, confidence, draw_position, duration, is_stable
        )
        
        # Handle special gestures with hold time
        self._handle_special_gestures(state)
        
        return state
    
    def _classify_gesture(
        self,
        finger_states: Dict[str, bool],
        landmarks: Dict[HandLandmark, Point],
        thumb_tip: Optional[Point],
        index_tip: Optional[Point],
        middle_tip: Optional[Point],
        ring_tip: Optional[Point],
        pinky_tip: Optional[Point]
    ) -> Tuple[Gesture, float]:
        """
        Classify the gesture based on finger states and positions.
        
        Returns:
            Tuple of (Gesture, confidence)
        """
        thumb = finger_states.get('thumb', False)
        index = finger_states.get('index', False)
        middle = finger_states.get('middle', False)
        ring = finger_states.get('ring', False)
        pinky = finger_states.get('pinky', False)
        
        # Count extended fingers
        extended_count = sum([thumb, index, middle, ring, pinky])
        
        # 1. Check for PINCH (thumb + index close together) - SUBMIT
        if thumb_tip and index_tip:
            pinch_distance = thumb_tip.distance_to(index_tip)
            if pinch_distance < self.PINCH_THRESHOLD:
                return Gesture.SUBMIT, 0.9
        
        # 2. Check for CLOSED FIST - CLEAR
        if extended_count == 0:
            # Verify all fingertips are close to palm
            wrist = landmarks.get(HandLandmark.WRIST)
            if wrist and index_tip:
                # Check if fingers are curled
                palm_center = landmarks.get(HandLandmark.MIDDLE_MCP)
                if palm_center and index_tip:
                    dist = index_tip.distance_to(palm_center)
                    if dist < self.FIST_THRESHOLD:
                        return Gesture.CLEAR, 0.85
            return Gesture.CLEAR, 0.7
        
        # 3. Check for UNDO (only pinky extended)
        if pinky and not thumb and not index and not middle and not ring:
            return Gesture.UNDO, 0.8
        
        # 4. Check for ERASE (index + middle + ring extended)
        if index and middle and ring and not pinky:
            return Gesture.ERASE, 0.85
        
        # 5. Check for PAUSE (index + middle extended, others down)
        if index and middle and not ring and not pinky:
            return Gesture.PAUSE, 0.9
        
        # 6. Check for DRAW (only index extended)
        if index and not middle and not ring and not pinky:
            return Gesture.DRAW, 0.95
        
        # Default: no recognized gesture
        return Gesture.NONE, 0.5
    
    def _smooth_gesture(self, gesture: Gesture) -> Gesture:
        """
        Apply temporal smoothing to gesture detection.
        Uses majority voting over recent frames.
        """
        self._gesture_history.append(gesture)
        
        if len(self._gesture_history) < self.smoothing_window // 2:
            return gesture
        
        # Count gesture occurrences
        gesture_counts = {}
        for g in self._gesture_history:
            gesture_counts[g] = gesture_counts.get(g, 0) + 1
        
        # Return most common gesture
        return max(gesture_counts, key=gesture_counts.get)
    
    def _smooth_position(self) -> Tuple[int, int]:
        """
        Smooth the drawing position using moving average.
        """
        if not self._position_history:
            return (0, 0)
        
        x = int(sum(p[0] for p in self._position_history) / len(self._position_history))
        y = int(sum(p[1] for p in self._position_history) / len(self._position_history))
        return (x, y)
    
    def _create_state(
        self,
        gesture: Gesture,
        confidence: float,
        position: Optional[Tuple[int, int]],
        duration: float = 0.0,
        is_stable: bool = False
    ) -> GestureState:
        """Create a GestureState object."""
        return GestureState(
            gesture=gesture,
            confidence=confidence,
            position=position,
            duration=duration,
            is_stable=is_stable
        )
    
    def _handle_special_gestures(self, state: GestureState):
        """
        Handle gestures that require hold time.
        Triggers callbacks for submit and clear after hold time.
        """
        if state.gesture == Gesture.SUBMIT:
            if state.duration >= self.SUBMIT_HOLD_TIME and not self._submit_triggered:
                self._submit_triggered = True
                if Gesture.SUBMIT in self._callbacks:
                    self._callbacks[Gesture.SUBMIT]()
        
        elif state.gesture == Gesture.CLEAR:
            if state.duration >= self.CLEAR_HOLD_TIME and not self._clear_triggered:
                self._clear_triggered = True
                if Gesture.CLEAR in self._callbacks:
                    self._callbacks[Gesture.CLEAR]()
    
    def register_callback(self, gesture: Gesture, callback: callable):
        """
        Register a callback for a specific gesture.
        
        Args:
            gesture: The gesture to trigger the callback
            callback: Function to call when gesture is triggered
        """
        self._callbacks[gesture] = callback
    
    def reset(self):
        """Reset the detector state."""
        self._gesture_history.clear()
        self._position_history.clear()
        self._current_gesture = Gesture.NONE
        self._gesture_start_time = 0.0
        self._submit_triggered = False
        self._clear_triggered = False
    
    def get_gesture_info(self, gesture: Gesture) -> dict:
        """
        Get information about a gesture.
        
        Returns:
            Dict with gesture name, description, and visualization
        """
        info = {
            Gesture.NONE: {
                'name': 'None',
                'description': 'No gesture detected',
                'icon': '❓'
            },
            Gesture.DRAW: {
                'name': 'Draw',
                'description': 'Index finger up - Draw on canvas',
                'icon': '✏️'
            },
            Gesture.PAUSE: {
                'name': 'Pause',
                'description': 'Index + Middle up - Pause drawing',
                'icon': '✋'
            },
            Gesture.ERASE: {
                'name': 'Erase',
                'description': 'Three fingers up - Erase mode',
                'icon': '🧹'
            },
            Gesture.CLEAR: {
                'name': 'Clear',
                'description': 'Closed fist (hold) - Clear canvas',
                'icon': '🗑️'
            },
            Gesture.SUBMIT: {
                'name': 'Submit',
                'description': 'Thumb + Index pinch (hold) - Submit to AI',
                'icon': '🚀'
            },
            Gesture.UNDO: {
                'name': 'Undo',
                'description': 'Pinky up only - Undo last stroke',
                'icon': '↩️'
            }
        }
        return info.get(gesture, info[Gesture.NONE])


def draw_gesture_ui(
    frame: np.ndarray,
    gesture_state: GestureState,
    detector: GestureDetector
) -> np.ndarray:
    """
    Draw gesture information on the frame.
    
    Args:
        frame: Image to draw on
        gesture_state: Current gesture state
        detector: GestureDetector instance for getting info
        
    Returns:
        Frame with gesture UI overlay
    """
    import cv2
    
    h, w = frame.shape[:2]
    
    # Get gesture info
    info = detector.get_gesture_info(gesture_state.gesture)
    
    # Draw gesture status box
    box_h = 80
    cv2.rectangle(frame, (10, h - box_h - 10), (250, h - 10), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, h - box_h - 10), (250, h - 10), (255, 255, 255), 2)
    
    # Gesture name
    color = (0, 255, 0) if gesture_state.is_stable else (0, 255, 255)
    cv2.putText(
        frame, f"{info['icon']} {info['name']}",
        (20, h - box_h + 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
    )
    
    # Confidence bar
    bar_width = int(200 * gesture_state.confidence)
    cv2.rectangle(frame, (20, h - 45), (20 + bar_width, h - 35), color, -1)
    cv2.rectangle(frame, (20, h - 45), (220, h - 35), (100, 100, 100), 1)
    
    # Duration for special gestures
    if gesture_state.gesture in [Gesture.SUBMIT, Gesture.CLEAR]:
        required_time = (detector.SUBMIT_HOLD_TIME if gesture_state.gesture == Gesture.SUBMIT
                        else detector.CLEAR_HOLD_TIME)
        progress = min(gesture_state.duration / required_time, 1.0)
        progress_width = int(200 * progress)
        cv2.rectangle(frame, (20, h - 25), (20 + progress_width, h - 15), (0, 200, 255), -1)
        cv2.rectangle(frame, (20, h - 25), (220, h - 15), (100, 100, 100), 1)
    
    return frame


if __name__ == "__main__":
    # Test gesture detection
    import cv2
    from camera import Camera
    from hand_tracking import HandTracker
    
    print("Testing Gesture Logic Module")
    print("=" * 40)
    print("Gestures:")
    print("  - Index finger up: DRAW")
    print("  - Index + Middle up: PAUSE")
    print("  - Three fingers up: ERASE")
    print("  - Closed fist (hold): CLEAR")
    print("  - Thumb + Index pinch (hold): SUBMIT")
    print("  - Pinky only: UNDO")
    print("Press 'q' to quit")
    
    with Camera() as cam:
        tracker = HandTracker(max_hands=1)
        detector = GestureDetector()
        
        while True:
            frame = cam.get_frame()
            if frame is None:
                continue
            
            # Process hand tracking
            hands = tracker.process(frame)
            
            # Detect gesture
            hand_data = hands[0] if hands else None
            gesture_state = detector.detect(hand_data)
            
            # Draw landmarks
            if hand_data:
                frame = tracker.draw_landmarks(frame, hand_data)
            
            # Draw gesture UI
            frame = draw_gesture_ui(frame, gesture_state, detector)
            
            # Draw position marker
            if gesture_state.position and gesture_state.gesture == Gesture.DRAW:
                cv2.circle(frame, gesture_state.position, 10, (0, 255, 255), -1)
            
            cv2.imshow("Gesture Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        tracker.release()
        cv2.destroyAllWindows()
