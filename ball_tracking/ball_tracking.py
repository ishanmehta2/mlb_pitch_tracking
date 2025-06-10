#!/usr/bin/env python
# ball_tracking.py
# -------------------------------------------------------------
#  Ball tracking and trajectory feature extraction for MLB pitches
# -------------------------------------------------------------

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple
import warnings

class BallTracker:
    """Ball tracking class for extracting trajectory features from pitch videos"""
    
    def __init__(self, 
                 min_area: int = 20, 
                 max_area: int = 500,
                 min_circularity: float = 0.4,
                 bg_history: int = 20,
                 bg_threshold: float = 25):
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.bg_history = bg_history
        self.bg_threshold = bg_threshold
    
    def track_ball_in_video(self, video_path: str) -> Optional[np.ndarray]:
        """Track ball positions throughout the video"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            warnings.warn(f"Could not open video file {video_path}")
            return None
        
        ball_positions = []
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.bg_history, 
            varThreshold=self.bg_threshold
        )
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            fg_mask = bg_subtractor.apply(frame)

            kernel = np.ones((3, 3), np.uint8)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

            ball_position = self._find_ball_in_frame(fg_mask)
            
            if ball_position is not None:
                cx, cy = ball_position
                ball_positions.append((cx, cy, frame_count))
        
        cap.release()
        return np.array(ball_positions) if ball_positions else None
    
    def _find_ball_in_frame(self, fg_mask: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find the most likely ball position in a single frame"""
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_ball = None
        max_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity > self.min_circularity:
                score = circularity * min(area / 100, 1.0)
                
                if score > max_score:
                    max_score = score
                    best_ball = contour
        
        if best_ball is not None:
            M = cv2.moments(best_ball)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
        
        return None

class TrajectoryFeatureExtractor:
    """Extract trajectory-based features from ball positions"""
    
    def __init__(self):
        self.feature_names = [
            'vertical_drop', 'horizontal_movement', 'avg_velocity', 'max_velocity',
            'curvature', 'curve_ratio', 'total_travel', 'trajectory_length',
            'velocity_variance', 'acceleration_avg', 'trajectory_angle',
            'smoothness'
        ]
    
    def extract_features(self, ball_positions: np.ndarray) -> Optional[Dict[str, float]]:
        """Extract comprehensive trajectory features"""
        if ball_positions is None or len(ball_positions) < 5:
            return None
            
        x = ball_positions[:, 0].astype(float)
        y = ball_positions[:, 1].astype(float)
        t = ball_positions[:, 2].astype(float)
        
        try:
            vertical_drop = y[-1] - y[0]
            horizontal_movement = x[-1] - x[0]

            velocities = self._calculate_velocities(x, y, t)
            avg_velocity = np.mean(velocities) if len(velocities) > 0 else 0
            max_velocity = np.max(velocities) if len(velocities) > 0 else 0
            velocity_variance = np.var(velocities) if len(velocities) > 0 else 0

            accelerations = self._calculate_accelerations(velocities, t)
            acceleration_avg = np.mean(np.abs(accelerations)) if len(accelerations) > 0 else 0

            curvature = self._calculate_curvature(x, y)
            curve_ratio = self._calculate_curve_ratio(x, y)
            total_travel = self._calculate_total_distance(x, y)
            trajectory_angle = self._calculate_trajectory_angle(x, y)
            smoothness = self._calculate_smoothness(x, y)
            
            features = {
                'vertical_drop': vertical_drop,
                'horizontal_movement': horizontal_movement,
                'avg_velocity': avg_velocity,
                'max_velocity': max_velocity,
                'curvature': curvature,
                'curve_ratio': curve_ratio,
                'total_travel': total_travel,
                'trajectory_length': len(ball_positions),
                'velocity_variance': velocity_variance,
                'acceleration_avg': acceleration_avg,
                'trajectory_angle': trajectory_angle,
                'smoothness': smoothness
            }
            
            return features
            
        except Exception as e:
            warnings.warn(f"Error extracting trajectory features: {e}")
            return None
    
    def _calculate_velocities(self, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Calculate frame-to-frame velocities"""
        if len(x) <= 1:
            return np.array([0])
        
        dx = np.diff(x)
        dy = np.diff(y)
        dt = np.diff(t)
        dt[dt == 0] = 1 
        
        return np.sqrt(dx**2 + dy**2) / dt
    
    def _calculate_accelerations(self, velocities: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Calculate accelerations from velocities"""
        if len(velocities) <= 1:
            return np.array([0])
        
        dv = np.diff(velocities)
        dt = np.diff(t[:-1]) 
        dt[dt == 0] = 1
        
        return dv / dt
    
    def _calculate_curvature(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate trajectory curvature using polynomial fit"""
        try:
            if len(x) < 3:
                return 0.0

            z = np.polyfit(x, y, min(2, len(x)-1))
            return abs(z[0]) if len(z) > 2 else 0.0
        except:
            return 0.0
    
    def _calculate_curve_ratio(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate ratio of actual path length to straight-line distance"""
        total_distance = self._calculate_total_distance(x, y)
        straight_line_distance = np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2)
        
        return total_distance / straight_line_distance if straight_line_distance > 0 else 1.0
    
    def _calculate_total_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate total path length"""
        return np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    
    def _calculate_trajectory_angle(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate overall trajectory angle in degrees"""
        if len(x) < 2:
            return 0.0
        
        delta_x = x[-1] - x[0]
        delta_y = y[-1] - y[0]
        
        return np.degrees(np.arctan2(delta_y, delta_x))
    
    def _calculate_smoothness(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate trajectory smoothness (inverse of jerk)"""
        if len(x) < 4:
            return 1.0

        dx2 = np.diff(x, n=2)
        dy2 = np.diff(y, n=2)

        jerk_x = np.diff(dx2)
        jerk_y = np.diff(dy2)

        jerk_magnitude = np.mean(np.sqrt(jerk_x**2 + jerk_y**2))

        return 1.0 / (1.0 + jerk_magnitude)

class TrajectoryNet(nn.Module):
    """Neural network for processing trajectory features"""
    
    def __init__(self, input_dim: int = 12, hidden_dim: int = 64, output_dim: int = 32):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        
        self.feature_names = [
            'vertical_drop', 'horizontal_movement', 'avg_velocity', 'max_velocity',
            'curvature', 'curve_ratio', 'total_travel', 'trajectory_length',
            'velocity_variance', 'acceleration_avg', 'trajectory_angle',
            'smoothness'
        ]
    
    def forward(self, x):
        return self.net(x)

def extract_trajectory_features_from_video(video_path: str) -> Optional[torch.Tensor]:
    """Main function to extract trajectory features from a video file"""
    tracker = BallTracker()
    extractor = TrajectoryFeatureExtractor()

    ball_positions = tracker.track_ball_in_video(video_path)
    
    if ball_positions is None:
        return torch.zeros(12)

    features_dict = extractor.extract_features(ball_positions)
    
    if features_dict is None:
        return torch.zeros(12)

    feature_vector = []
    for feature_name in extractor.feature_names:
        feature_vector.append(features_dict.get(feature_name, 0.0))
    
    return torch.tensor(feature_vector, dtype=torch.float32)

def normalize_trajectory_features(features: torch.Tensor) -> torch.Tensor:
    """Normalize trajectory features for better training stability"""

    normalization_factors = torch.tensor([
        100.0,  # vertical_drop
        100.0,  # horizontal_movement  
        20.0,   # avg_velocity
        50.0,   # max_velocity
        0.01,   # curvature
        2.0,    # curve_ratio
        500.0,  # total_travel
        50.0,   # trajectory_length
        100.0,  # velocity_variance
        10.0,   # acceleration_avg
        180.0,  # trajectory_angle
        1.0     # smoothness
    ])
    
    return features / normalization_factors

if __name__ == "__main__":
    test_video = "test_pitch.mp4" 
    features = extract_trajectory_features_from_video(test_video)
    
    if features is not None:
        print("Extracted trajectory features:")
        feature_names = [
            'vertical_drop', 'horizontal_movement', 'avg_velocity', 'max_velocity',
            'curvature', 'curve_ratio', 'total_travel', 'trajectory_length',
            'velocity_variance', 'acceleration_avg', 'trajectory_angle',
            'smoothness'
        ]
        
        for name, value in zip(feature_names, features):
            print(f"{name}: {value:.4f}")
    else:
        print("Failed to extract trajectory features")
