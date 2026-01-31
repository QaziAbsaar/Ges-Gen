"""
Camera Module - Webcam Stream Handler
======================================
Handles webcam capture, frame processing, and video stream management.
Designed for real-time performance with configurable resolution and FPS.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Generator
import threading
import time


class Camera:
    """
    Webcam stream handler with threading support for smooth frame capture.
    
    Attributes:
        camera_id: Index of the camera device (default 0)
        width: Frame width in pixels
        height: Frame height in pixels
        fps: Target frames per second
    """
    
    def __init__(
        self,
        camera_id: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30
    ):
        """
        Initialize the camera with specified parameters.
        
        Args:
            camera_id: Camera device index
            width: Desired frame width
            height: Desired frame height
            fps: Target frame rate
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        
        # Video capture object
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Threading components for non-blocking capture
        self._frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Performance metrics
        self._actual_fps = 0.0
        self._frame_count = 0
        self._start_time = 0.0
    
    def start(self) -> bool:
        """
        Start the camera capture.
        
        Returns:
            True if camera started successfully, False otherwise
        """
        # Initialize video capture with DirectShow backend on Windows
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            print(f"[ERROR] Failed to open camera {self.camera_id}")
            return False
        
        # Configure camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Set buffer size to 1 for minimum latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Get actual resolution (may differ from requested)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"[INFO] Camera started: {self.width}x{self.height} @ {self.fps}fps")
        
        # Start capture thread
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        
        return True
    
    def _capture_loop(self):
        """
        Internal capture loop running in separate thread.
        Continuously captures frames to ensure we always have the latest frame.
        """
        while self._running:
            ret, frame = self.cap.read()
            
            if ret:
                # Flip horizontally for mirror effect (more intuitive for user)
                frame = cv2.flip(frame, 1)
                
                with self._frame_lock:
                    self._frame = frame
                    self._frame_count += 1
            else:
                time.sleep(0.001)  # Small delay to prevent busy waiting
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the latest captured frame.
        
        Returns:
            Tuple of (success: bool, frame: np.ndarray or None)
        """
        with self._frame_lock:
            if self._frame is None:
                return False, None
            # Return a copy to prevent race conditions
            return True, self._frame.copy()
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest frame (convenience method).
        
        Returns:
            Frame as numpy array or None if no frame available
        """
        ret, frame = self.read()
        return frame if ret else None
    
    def get_fps(self) -> float:
        """
        Calculate and return actual FPS.
        
        Returns:
            Current frames per second
        """
        elapsed = time.time() - self._start_time
        if elapsed > 0:
            self._actual_fps = self._frame_count / elapsed
        return self._actual_fps
    
    def get_resolution(self) -> Tuple[int, int]:
        """
        Get current camera resolution.
        
        Returns:
            Tuple of (width, height)
        """
        return self.width, self.height
    
    def stop(self):
        """Stop the camera capture and release resources."""
        self._running = False
        
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        print("[INFO] Camera stopped")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
    
    def frame_generator(self) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields frames continuously.
        
        Yields:
            Frames as numpy arrays
        """
        while self._running:
            frame = self.get_frame()
            if frame is not None:
                yield frame
            else:
                time.sleep(0.001)


# Utility functions
def list_available_cameras(max_cameras: int = 10) -> list:
    """
    List all available camera devices.
    
    Args:
        max_cameras: Maximum number of cameras to check
        
    Returns:
        List of available camera indices
    """
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


if __name__ == "__main__":
    # Test camera module
    print("Testing Camera Module")
    print("=" * 40)
    
    available = list_available_cameras()
    print(f"Available cameras: {available}")
    
    if available:
        with Camera(camera_id=available[0]) as cam:
            print("Press 'q' to quit")
            
            start = time.time()
            while time.time() - start < 10:  # Run for 10 seconds
                frame = cam.get_frame()
                if frame is not None:
                    # Display FPS on frame
                    fps = cam.get_fps()
                    cv2.putText(
                        frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                    )
                    cv2.imshow("Camera Test", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cv2.destroyAllWindows()
    else:
        print("No cameras found!")
