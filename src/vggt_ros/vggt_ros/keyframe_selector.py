import cv2
import numpy as np

class KeyframeSelector:
    def __init__(self, window_size=8, min_parallax=10.0):
        """
        Args:
            window_size (int): Number of keyframes to keep in the window.
            min_parallax (float): Minimum average pixel displacement (optical flow) to select a new keyframe.
                                  Values <= 1 are treated as a fraction of the image diagonal (e.g. 0.1 = 10%).
        """
        self.window_size = window_size
        self.min_parallax = min_parallax
        self.keyframes = [] # List of (cv_image, header)
        self.last_keyframe_gray = None
        
    def process_frame(self, cv_image, header):
        """
        Process a new frame from the stream.
        Returns:
            bool: True if a new keyframe was added, False otherwise.
        """
        # Convert to grayscale for flow/feature calculation
        if len(cv_image.shape) == 3:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = cv_image
        
        if not self.keyframes:
            # First frame is always a keyframe
            self.add_keyframe(cv_image, header, gray)
            return True
            
        # Calculate parallax relative to the last keyframe
        flow_magnitude = self.calculate_parallax(self.last_keyframe_gray, gray)
        threshold = self._parallax_threshold(cv_image.shape)

        if flow_magnitude >= threshold:
            self.add_keyframe(cv_image, header, gray)
            return True # New keyframe added
            
        return False # Frame skipped (not enough parallax)

    def _parallax_threshold(self, image_shape):
        height, width = image_shape[0], image_shape[1]
        if self.min_parallax <= 1.0:
            diag = (height**2 + width**2) ** 0.5
            return self.min_parallax * diag
        return self.min_parallax

    def add_keyframe(self, cv_image, header, gray):
        self.keyframes.append((cv_image, header))
        self.last_keyframe_gray = gray
        
        # Maintain window size
        if len(self.keyframes) > self.window_size:
            self.keyframes.pop(0)
            
    def calculate_parallax(self, img1, img2):
        # Simple approach: Optical Flow using Lucas-Kanade
        # 1. Detect features in the previous keyframe
        p0 = cv2.goodFeaturesToTrack(img1, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        
        if p0 is None or len(p0) == 0:
            # If no features found, assume large change or empty image, force keyframe? 
            # Or return 0 to skip? Let's return infinity to force update if scene changed drastically
            return float('inf')
            
        # 2. Calculate optical flow to the current frame
        p1, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None)
        
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
            
            if len(good_new) == 0:
                return float('inf')
                
            # Calculate distances
            dist = np.linalg.norm(good_new - good_old, axis=1)
            return np.mean(dist)
            
        return float('inf')
        
    def get_window(self):
        # Return lists of images and headers
        images = [k[0] for k in self.keyframes]
        headers = [k[1] for k in self.keyframes]
        return images, headers
        
    def is_full(self):
        return len(self.keyframes) >= self.window_size
    
    def clear(self):
        self.keyframes = []
        self.last_keyframe_gray = None
