#!/usr/bin/env python
# advanced_ball_tracking.py
# -------------------------------------------------------------
# Advanced ball tracking with multiple detection strategies
# Combines YOLO, optical flow, and motion detection
# -------------------------------------------------------------

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import os
from collections import defaultdict, deque
from scipy import ndimage
from sklearn.cluster import DBSCAN

# Import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLO available for advanced tracking")
except ImportError:
    YOLO_AVAILABLE = False
    print("❌ YOLO not available")

class AdvancedBallTracker:
    def __init__(self):
        self.yolo_model = None
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False, varThreshold=50, history=10
        )
        self.optical_flow_params = dict(
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7
        )
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Tracking state
        self.previous_frame = None
        self.ball_trajectory = deque(maxlen=30)  # Store last 30 detections
        self.candidate_tracks = []
        
        # Initialize YOLO
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO("yolov8x.pt")
                self.yolo_model.classes = [32, 37]  # sports ball (32) + baseball/frisbee (37)
                self.yolo_model.conf = 0.1  # Lower confidence threshold
                self.yolo_model.iou = 0.3
                print("✅ Advanced YOLO tracking initialized")
            except Exception as e:
                print(f"⚠️  YOLO initialization failed: {e}")
                self.yolo_model = None
    
    def detect_moving_objects(self, frame):
        """Detect moving objects using background subtraction"""
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        moving_objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by size (ball should be small but not tiny)
            if 10 < area < 2000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Filter by aspect ratio (ball should be roughly circular)
                if 0.5 < aspect_ratio < 2.0:
                    # Calculate center and add confidence based on circularity
                    cx, cy = x + w//2, y + h//2
                    
                    # Calculate circularity
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        confidence = min(circularity * 2, 1.0)  # Scale to 0-1
                        
                        moving_objects.append({
                            'center': (cx, cy),
                            'bbox': (x, y, x+w, y+h),
                            'area': area,
                            'confidence': confidence,
                            'method': 'motion'
                        })
        
        return moving_objects, fg_mask
    
    def detect_circular_objects(self, frame):
        """Detect circular objects using Hough circles"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=3,
            maxRadius=50
        )
        
        circular_objects = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # Calculate confidence based on circle properties
                confidence = min(r / 20.0, 1.0)  # Prefer medium-sized circles
                
                circular_objects.append({
                    'center': (x, y),
                    'bbox': (x-r, y-r, x+r, y+r),
                    'radius': r,
                    'confidence': confidence,
                    'method': 'hough'
                })
        
        return circular_objects
    
    def detect_with_yolo(self, frame):
        """Enhanced YOLO detection with multiple crops"""
        if self.yolo_model is None:
            return []
        
        yolo_detections = []
        h, w = frame.shape[:2]
        
        # Try multiple crop regions to catch ball trajectory
        crop_regions = [
            (0, int(h*0.7)),      # Top 70%
            (0, int(h*0.85)),     # Top 85% 
            (int(h*0.1), h),      # Bottom 90%
            (0, h)                # Full frame
        ]
        
        for top, bottom in crop_regions:
            cropped = frame[top:bottom, :]
            
            try:
                results = self.yolo_model.predict(cropped, conf=0.1, verbose=False)
                
                if len(results) > 0 and len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    cls_np = results[0].boxes.cls.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()
                    
                    for i, cls_id in enumerate(cls_np):
                        if int(cls_id) in [32, 37]:  # Ball classes
                            x1, y1, x2, y2 = boxes[i]
                            
                            # Adjust coordinates back to full frame
                            y1 += top
                            y2 += top
                            
                            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                            
                            yolo_detections.append({
                                'center': (cx, cy),
                                'bbox': (x1, y1, x2, y2),
                                'confidence': confs[i],
                                'method': 'yolo',
                                'class_id': int(cls_id)
                            })
            except Exception as e:
                continue
        
        return yolo_detections
    
    def track_with_optical_flow(self, frame):
        """Track previous detections using optical flow"""
        if self.previous_frame is None or len(self.ball_trajectory) == 0:
            return []
        
        gray_prev = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get last few ball positions for tracking
        tracked_objects = []
        if len(self.ball_trajectory) > 0:
            recent_positions = list(self.ball_trajectory)[-3:]  # Last 3 positions
            
            for pos_data in recent_positions:
                if 'center' in pos_data:
                    old_points = np.array([[pos_data['center']]], dtype=np.float32)
                    
                    # Calculate optical flow
                    new_points, status, error = cv2.calcOpticalFlowPyrLK(
                        gray_prev, gray_curr, old_points, None, **self.lk_params
                    )
                    
                    if status[0][0] == 1 and error[0][0] < 50:  # Good tracking
                        cx, cy = new_points[0][0]
                        
                        # Predict confidence based on tracking quality
                        confidence = max(0.3, 1.0 - error[0][0] / 50.0)
                        
                        tracked_objects.append({
                            'center': (cx, cy),
                            'bbox': (cx-10, cy-10, cx+10, cy+10),
                            'confidence': confidence,
                            'method': 'optical_flow'
                        })
        
        return tracked_objects
    
    def filter_detections_by_physics(self, detections):
        """Filter detections based on baseball physics"""
        if len(self.ball_trajectory) == 0:
            return detections
        
        filtered = []
        last_pos = self.ball_trajectory[-1]['center']
        
        for detection in detections:
            cx, cy = detection['center']
            
            # Calculate distance from last known position
            distance = np.sqrt((cx - last_pos[0])**2 + (cy - last_pos[1])**2)
            
            # Filter by reasonable movement (adjust based on frame rate)
            max_movement = 100  # pixels per frame
            if distance < max_movement:
                # Boost confidence for reasonable movements
                physics_confidence = max(0.1, 1.0 - distance / max_movement)
                detection['confidence'] *= physics_confidence
                filtered.append(detection)
        
        return filtered
    
    def combine_and_rank_detections(self, all_detections):
        """Combine detections from all methods and rank by confidence"""
        if not all_detections:
            return []
        
        # Cluster nearby detections
        centers = np.array([det['center'] for det in all_detections])
        
        if len(centers) > 1:
            # Use DBSCAN to cluster nearby detections
            clustering = DBSCAN(eps=30, min_samples=1).fit(centers)
            labels = clustering.labels_
            
            # Merge clustered detections
            merged_detections = []
            for label in set(labels):
                cluster_mask = labels == label
                cluster_detections = [det for i, det in enumerate(all_detections) if cluster_mask[i]]
                
                # Merge by weighted average
                total_confidence = sum(det['confidence'] for det in cluster_detections)
                if total_confidence > 0:
                    weighted_x = sum(det['center'][0] * det['confidence'] for det in cluster_detections) / total_confidence
                    weighted_y = sum(det['center'][1] * det['confidence'] for det in cluster_detections) / total_confidence
                    
                    # Combine methods
                    methods = [det['method'] for det in cluster_detections]
                    combined_method = '+'.join(set(methods))
                    
                    merged_detections.append({
                        'center': (weighted_x, weighted_y),
                        'confidence': total_confidence / len(cluster_detections),
                        'method': combined_method,
                        'num_sources': len(cluster_detections)
                    })
            
            all_detections = merged_detections
        
        # Sort by confidence
        all_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return all_detections
    
    def track_frame(self, frame):
        """Main tracking function - combines all methods"""
        all_detections = []
        
        # Method 1: YOLO detection
        yolo_detections = self.detect_with_yolo(frame)
        all_detections.extend(yolo_detections)
        
        # Method 2: Motion detection
        motion_detections, fg_mask = self.detect_moving_objects(frame)
        all_detections.extend(motion_detections)
        
        # Method 3: Circular object detection
        circular_detections = self.detect_circular_objects(frame)
        all_detections.extend(circular_detections)
        
        # Method 4: Optical flow tracking
        flow_detections = self.track_with_optical_flow(frame)
        all_detections.extend(flow_detections)
        
        # Filter by physics constraints
        if len(self.ball_trajectory) > 0:
            all_detections = self.filter_detections_by_physics(all_detections)
        
        # Combine and rank detections
        final_detections = self.combine_and_rank_detections(all_detections)
        
        # Store best detection in trajectory
        if final_detections:
            best_detection = final_detections[0]
            self.ball_trajectory.append(best_detection)
        
        # Update previous frame
        self.previous_frame = frame.copy()
        
        return final_detections, fg_mask if 'fg_mask' in locals() else None

def advanced_ball_tracking_diagnostic(video_path, output_dir="advanced_tracking_proof"):
    """Advanced diagnostic with multiple tracking methods"""
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize tracker
    tracker = AdvancedBallTracker()
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Video info: {frame_count} frames, {width}x{height}, {fps:.1f} FPS")
    
    # Tracking data storage
    all_trajectories = []
    detection_methods = defaultdict(int)
    annotated_frames = []
    frame_idx = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            original_frame = frame.copy()
            
            # Track ball in this frame
            detections, fg_mask = tracker.track_frame(frame)
            
            # Annotate frame
            if detections:
                for i, detection in enumerate(detections[:3]):  # Show top 3 detections
                    cx, cy = detection['center']
                    conf = detection['confidence']
                    method = detection['method']
                    
                    # Color by method
                    colors = {
                        'yolo': (0, 255, 0),
                        'motion': (255, 0, 0), 
                        'hough': (0, 0, 255),
                        'optical_flow': (255, 255, 0)
                    }
                    color = colors.get(method.split('+')[0], (128, 128, 128))
                    
                    # Draw detection
                    cv2.circle(original_frame, (int(cx), int(cy)), 8, color, -1)
                    cv2.circle(original_frame, (int(cx), int(cy)), 15, color, 2)
                    
                    # Add text
                    text = f"{method}: {conf:.2f}"
                    cv2.putText(original_frame, text, (int(cx)+20, int(cy)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Store detection data
                    if i == 0:  # Only store best detection
                        all_trajectories.append({
                            'frame': frame_idx,
                            'center': (cx, cy),
                            'confidence': conf,
                            'method': method
                        })
                        detection_methods[method] += 1
                
                # Draw trajectory
                if len(tracker.ball_trajectory) > 1:
                    points = [det['center'] for det in tracker.ball_trajectory]
                    for j in range(1, len(points)):
                        pt1 = (int(points[j-1][0]), int(points[j-1][1]))
                        pt2 = (int(points[j][0]), int(points[j][1]))
                        cv2.line(original_frame, pt1, pt2, (255, 255, 255), 2)
            
            # Add frame info
            info_text = f"Frame: {frame_idx} | Detections: {len(detections)} | Total tracked: {len(all_trajectories)}"
            cv2.putText(original_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Store annotated frames
            if frame_idx % 10 == 0 or detections:  # Every 10th frame or detection frames
                annotated_frames.append((frame_idx, original_frame.copy()))
            
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                print(f"   Processed {frame_idx}/{frame_count} frames...")
    
    except Exception as e:
        print(f"❌ Error during tracking: {e}")
    finally:
        cap.release()
    
    # Analysis
    detection_rate = len(all_trajectories) / frame_count if frame_count > 0 else 0
    
    print(f"\n📊 ADVANCED TRACKING RESULTS:")
    print(f"   Total frames: {frame_count}")
    print(f"   Frames with ball detected: {len(all_trajectories)}")
    print(f"   Detection rate: {detection_rate:.1%}")
    
    print(f"\n🔧 DETECTION METHODS:")
    for method, count in detection_methods.items():
        print(f"   {method:<20}: {count:4d} detections ({count/len(all_trajectories)*100:.1f}%)")
    
    # Create visualizations
    if all_trajectories:
        create_advanced_visualizations(all_trajectories, annotated_frames, output_dir, video_path)
        
        # Calculate trajectory features
        features = calculate_advanced_trajectory_features(all_trajectories, frame_count, width, height)
        
        return {
            'trajectories': all_trajectories,
            'features': features,
            'detection_rate': detection_rate,
            'method_breakdown': dict(detection_methods)
        }
    else:
        print("❌ No ball detections found with advanced tracking!")
        return None

def calculate_advanced_trajectory_features(trajectories, total_frames, frame_width, frame_height):
    """Calculate enhanced trajectory features"""
    
    if len(trajectories) < 2:
        return torch.zeros(15, dtype=torch.float32)  # Expanded to 15 features
    
    # Extract data
    frames = np.array([t['frame'] for t in trajectories])
    x_coords = np.array([t['center'][0] / frame_width for t in trajectories])
    y_coords = np.array([t['center'][1] / frame_height for t in trajectories])
    confidences = np.array([t['confidence'] for t in trajectories])
    
    features = []
    
    # 1. Position statistics
    features.extend([
        np.mean(x_coords),
        np.mean(y_coords),
        np.std(x_coords),
        np.std(y_coords),
        np.min(x_coords),
        np.max(x_coords),
        np.min(y_coords),
        np.max(y_coords)
    ])
    
    # 2. Temporal statistics
    features.append(len(trajectories) / max(total_frames, 1))  # detection rate
    features.append(np.mean(confidences))  # avg confidence
    
    # 3. Velocity features
    if len(frames) > 1:
        frame_diffs = np.diff(frames)
        frame_diffs[frame_diffs == 0] = 1
        
        x_velocity = np.diff(x_coords) / frame_diffs
        y_velocity = np.diff(y_coords) / frame_diffs
        speed = np.sqrt(x_velocity**2 + y_velocity**2)
        
        features.extend([
            np.mean(speed),
            np.std(speed),
            np.mean(y_velocity)  # Vertical velocity (important for pitch drop)
        ])
    else:
        features.extend([0.0, 0.0, 0.0])
    
    # 4. Trajectory shape
    if len(x_coords) >= 3:
        # Curvature
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)
        if len(dx) >= 2:
            d2x = np.diff(dx)
            d2y = np.diff(dy)
            curvature = np.mean(np.sqrt(d2x**2 + d2y**2))
        else:
            curvature = 0.0
        features.append(curvature)
    else:
        features.append(0.0)
    
    # 5. Endpoint displacement
    if len(x_coords) > 0:
        features.append(y_coords[-1] - y_coords[0])  # vertical displacement
    else:
        features.append(0.0)
    
    # Ensure exactly 15 features
    features = features[:15]
    while len(features) < 15:
        features.append(0.0)
    
    return torch.tensor(features, dtype=torch.float32)

def create_advanced_visualizations(trajectories, annotated_frames, output_dir, video_path):
    """Create advanced visualization plots"""
    
    # Save annotated frames
    print(f"📸 Saving {min(len(annotated_frames), 15)} annotated frames...")
    for i, (frame_idx, frame) in enumerate(annotated_frames[:15]):
        frame_path = Path(output_dir) / f"frame_{frame_idx:04d}.jpg"
        cv2.imwrite(str(frame_path), frame)
    
    # Create trajectory analysis
    if len(trajectories) > 1:
        plt.figure(figsize=(20, 12))
        
        frames = [t['frame'] for t in trajectories]
        x_coords = [t['center'][0] for t in trajectories]
        y_coords = [t['center'][1] for t in trajectories]
        confidences = [t['confidence'] for t in trajectories]
        methods = [t['method'] for t in trajectories]
        
        # 1. 2D trajectory with method colors
        plt.subplot(2, 4, 1)
        method_colors = {
            'yolo': 'green',
            'motion': 'red',
            'hough': 'blue', 
            'optical_flow': 'orange'
        }
        
        for method, color in method_colors.items():
            method_mask = [method in m for m in methods]
            if any(method_mask):
                method_x = [x for i, x in enumerate(x_coords) if method_mask[i]]
                method_y = [y for i, y in enumerate(y_coords) if method_mask[i]]
                plt.scatter(method_x, method_y, c=color, label=method, s=30, alpha=0.7)
        
        plt.plot(x_coords, y_coords, 'k-', alpha=0.3, linewidth=1)
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Y Position (pixels)')
        plt.title('Ball Trajectory by Detection Method')
        plt.legend()
        plt.gca().invert_yaxis()
        
        # 2. X position over time
        plt.subplot(2, 4, 2)
        plt.plot(frames, x_coords, 'b-o', markersize=3)
        plt.xlabel('Frame')
        plt.ylabel('X Position')
        plt.title('Horizontal Movement')
        plt.grid(True, alpha=0.3)
        
        # 3. Y position over time
        plt.subplot(2, 4, 3)
        plt.plot(frames, y_coords, 'r-o', markersize=3)
        plt.xlabel('Frame')
        plt.ylabel('Y Position')
        plt.title('Vertical Movement')
        plt.grid(True, alpha=0.3)
        
        # 4. Confidence over time
        plt.subplot(2, 4, 4)
        plt.plot(frames, confidences, 'g-o', markersize=3)
        plt.xlabel('Frame')
        plt.ylabel('Confidence')
        plt.title('Detection Confidence')
        plt.grid(True, alpha=0.3)
        
        # 5. Speed analysis
        plt.subplot(2, 4, 5)
        if len(frames) > 1:
            speeds = []
            for i in range(1, len(frames)):
                dx = x_coords[i] - x_coords[i-1]
                dy = y_coords[i] - y_coords[i-1]
                speed = np.sqrt(dx**2 + dy**2)
                speeds.append(speed)
            
            plt.plot(frames[1:], speeds, 'm-o', markersize=3)
            plt.xlabel('Frame')
            plt.ylabel('Speed (pixels/frame)')
            plt.title('Ball Speed')
            plt.grid(True, alpha=0.3)
        
        # 6. Method distribution
        plt.subplot(2, 4, 6)
        method_counts = defaultdict(int)
        for method in methods:
            method_counts[method] += 1
        
        plt.pie(method_counts.values(), labels=method_counts.keys(), autopct='%1.1f%%')
        plt.title('Detection Method Distribution')
        
        # 7. Detection gaps
        plt.subplot(2, 4, 7)
        if len(frames) > 1:
            gaps = np.diff(frames)
            plt.hist(gaps, bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('Frame Gap Size')
            plt.ylabel('Frequency')
            plt.title('Detection Consistency')
            plt.grid(True, alpha=0.3)
        
        # 8. Trajectory density heatmap
        plt.subplot(2, 4, 8)
        plt.hist2d(x_coords, y_coords, bins=20, cmap='Blues')
        plt.colorbar(label='Detection Density')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Ball Position Heatmap')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        
        plot_path = Path(output_dir) / f"advanced_trajectory_analysis_{Path(video_path).stem}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Advanced trajectory analysis saved to: {plot_path}")

def test_advanced_tracking(video_dir="no_contact_pitches", num_videos=5):
    """Test advanced tracking on multiple videos"""
    
    print(f"🚀 TESTING ADVANCED BALL TRACKING")
    print("=" * 60)
    
    video_dir = Path(video_dir)
    if not video_dir.exists():
        print(f"❌ Video directory not found: {video_dir}")
        return
    
    video_files = list(video_dir.glob("*.mp4"))[:num_videos]
    
    if not video_files:
        print(f"❌ No video files found in {video_dir}")
        return
    
    print(f"📹 Testing {len(video_files)} videos with advanced tracking...")
    
    results = []
    
    for i, video_path in enumerate(video_files):
        print(f"\n{'='*60}")
        print(f"🎬 Video {i+1}/{len(video_files)}: {video_path.name}")
        print(f"{'='*60}")
        
        result = advanced_ball_tracking_diagnostic(
            video_path,
            output_dir=f"advanced_tracking_proof/video_{i+1}_{video_path.stem}"
        )
        
        if result:
            results.append({
                'video': video_path.name,
                'detection_rate': result['detection_rate'],
                'method_breakdown': result['method_breakdown'],
                'features': result['features']
            })
    
    # Summary
    if results:
        print(f"\n{'='*60}")
        print(f"📊 ADVANCED TRACKING SUMMARY")
        print(f"{'='*60}")
        
        detection_rates = [r['detection_rate'] for r in results]
        print(f"📈 Detection Rate Statistics:")
        print(f"   Average: {np.mean(detection_rates):.1%}")
        print(f"   Range: {np.min(detection_rates):.1%} - {np.max(detection_rates):.1%}")
        print(f"   Median: {np.median(detection_rates):.1%}")
        
        print(f"\n📋 Per-Video Results:")
        for r in results:
            print(f"   {r['video']:<30}: {r['detection_rate']:6.1%}")
        
        # Method effectiveness
        all_methods = defaultdict(int)
        for r in results:
            for method, count in r['method_breakdown'].items():
                all_methods[method] += count
        
        total_detections = sum(all_methods.values())
        print(f"\n🔧 Method Effectiveness (Total: {total_detections} detections):")
        for method, count in sorted(all_methods.items(), key=lambda x: x[1], reverse=True):
            print(f"   {method:<20}: {count:4d} ({count/total_detections*100:.1f}%)")

def main():
    """Main advanced diagnostic function"""
    print("🚀 ADVANCED BALL TRACKING DIAGNOSTIC")
    print("=" * 60)
    
    test_advanced_tracking(video_dir="no_contact_pitches", num_videos=5)
    
    print(f"\n✅ ADVANCED DIAGNOSTIC COMPLETE!")
    print(f"📁 Check 'advanced_tracking_proof/' folder for results")

if __name__ == "__main__":
    main()