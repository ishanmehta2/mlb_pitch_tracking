import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from collections import Counter

def process_video(video_path):
    """Process video to track the baseball and return positions"""
    # Read video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
    
    ball_positions = []
    
    # Background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=25)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find circular objects (potential baseballs)
        best_ball = None
        max_circularity = 0
        
        for contour in contours:
            # Filter by size
            area = cv2.contourArea(contour)
            if area < 20 or area > 500:  # Adjust based on your video
                continue
                
            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity > max_circularity and circularity > 0.5:
                max_circularity = circularity
                best_ball = contour
        
        if best_ball is not None:
            # Get center of the ball
            M = cv2.moments(best_ball)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                ball_positions.append((cx, cy, frame_count))
    
    cap.release()
    return np.array(ball_positions) if ball_positions else None

def extract_features(ball_positions):
    """Extract trajectory features from ball positions"""
    if ball_positions is None or len(ball_positions) < 5:
        return None
        
    # Extract x, y coordinates and frame numbers
    x = ball_positions[:, 0]
    y = ball_positions[:, 1]
    t = ball_positions[:, 2]
    
    # Calculate velocities (pixels per frame)
    if len(x) > 1:
        dx = np.diff(x)
        dy = np.diff(y)
        dt = np.diff(t)
        velocities = np.sqrt(dx**2 + dy**2) / dt
    else:
        velocities = np.array([0])
    
    # Fit polynomial to trajectory (y as function of x)
    try:
        # 2nd degree polynomial fit for trajectory
        z = np.polyfit(x, y, 2)
        polynomial = np.poly1d(z)
        
        # Calculate fitted y values
        y_fit = polynomial(x)
        
        # Calculate curvature
        curvature = abs(z[0])
        
        # Calculate additional metrics
        total_distance = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
        straight_line_distance = np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2)
        curve_ratio = total_distance / straight_line_distance if straight_line_distance > 0 else 1
        
        # Extract features
        features = {
            'vertical_drop': y[-1] - y[0],
            'horizontal_movement': x[-1] - x[0],
            'avg_velocity': np.mean(velocities),
            'max_velocity': np.max(velocities),
            'curvature': curvature,
            'curve_ratio': curve_ratio,
            'total_travel': total_distance,
            'final_x': x[-1],
            'final_y': y[-1],
            'start_x': x[0],
            'start_y': y[0]
        }
        
        return features
    
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def predict_pitch_type(features):
    """Predict pitch type based on extracted features"""
    if not features:
        return "unknown"
    
    # Extract key metrics
    curvature = features['curvature']
    horizontal_movement = abs(features['horizontal_movement'])
    vertical_drop = features['vertical_drop']
    curve_ratio = features['curve_ratio']
    
    # Simple rule-based classification
    # These thresholds are approximations and might need adjustment
    
    # Curveball - high curvature and significant vertical drop
    if curvature > 0.005 and vertical_drop > 80:
        return "curveball"
    
    # Slider - moderate curvature with significant horizontal movement
    elif (curvature > 0.002 and horizontal_movement > 70) or (curvature > 0.0015 and horizontal_movement > 100):
        return "slider"
    
    # Sinker - moderate downward and horizontal movement
    elif vertical_drop > 60 and horizontal_movement > 40 and curvature < 0.003:
        return "sinker"
    
    # Changeup - less velocity, moderate movement
    elif features['avg_velocity'] < 12 and curve_ratio < 1.15:
        return "changeup"
    
    # Knucklecurve - specific combination of movement patterns
    elif curvature > 0.004 and vertical_drop > 60 and horizontal_movement > 50:
        return "knucklecurve"
    
    # Fastball - high velocity, relatively straight path
    else:
        return "fastball"

def analyze_pitch(video_path):
    """Analyze a pitch video and return the predicted pitch type"""
    # Track the ball
    ball_positions = process_video(video_path)
    
    if ball_positions is not None and len(ball_positions) > 0:
        # Extract features
        features = extract_features(ball_positions)
        
        if features:
            # Predict pitch type
            pitch_type = predict_pitch_type(features)
            return pitch_type
    
    return "unknown"

def load_pitch_data(json_path):
    """Load pitch data from the MLB YouTube JSON file"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return {}

def extract_video_id_from_filename(filename):
    """Extract the video ID from the filename (remove extension)"""
    return os.path.splitext(filename)[0]

def main():
    """Main function to process videos and evaluate accuracy"""
    # Define paths
    data_dir = "baseline_data"
    json_path = "data/mlb-youtube-segmented.json"
    
    # Load pitch data
    pitch_data = load_pitch_data(json_path)
    
    if not pitch_data:
        print("No pitch data found. Exiting.")
        return
    
    # Get list of video files
    video_files = [f for f in os.listdir(data_dir) if f.endswith(('.mp4', '.mov'))]
    
    if not video_files:
        print(f"No video files found in {data_dir}")
        return
    
    results = []
    correct_predictions = 0
    total_predictions = 0
    
    # Process each video
    for filename in video_files:
        video_path = os.path.join(data_dir, filename)
        
        # Extract video ID from filename
        video_id = extract_video_id_from_filename(filename)
        
        # Get ground truth type from JSON
        true_type = "unknown"
        if video_id in pitch_data:
            if "type" in pitch_data[video_id]:
                true_type = pitch_data[video_id]["type"]
        
        if true_type == "unknown":
            print(f"Skipping {filename} - no ground truth type found in JSON")
            continue
        
        # Predict pitch type
        predicted_type = analyze_pitch(video_path)
        
        # Calculate accuracy
        is_correct = predicted_type == true_type
        if is_correct:
            correct_predictions += 1
        total_predictions += 1
        
        # Store results
        results.append({
            "filename": filename,
            "true_type": true_type,
            "predicted_type": predicted_type,
            "correct": is_correct
        })
        
        # Print individual result
        print(f"Video: {filename} | True: {true_type} | Predicted: {predicted_type} | Correct: {is_correct}")
    
    # Calculate and print overall accuracy
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions * 100
        print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")
        
        # Print confusion matrix
        print("\nPrediction Distribution:")
        true_labels = [r["true_type"] for r in results]
        pred_labels = [r["predicted_type"] for r in results]
        
        # Count occurrences of each pitch type
        true_counts = Counter(true_labels)
        pred_counts = Counter(pred_labels)
        
        print("\nGround Truth Distribution:")
        for pitch_type, count in true_counts.items():
            print(f"{pitch_type}: {count} pitches")
        
        print("\nPredicted Distribution:")
        for pitch_type, count in pred_counts.items():
            print(f"{pitch_type}: {count} pitches")
        
        # Calculate type-specific accuracy
        print("\nType-Specific Accuracy:")
        for pitch_type in set(true_labels):
            type_results = [r for r in results if r["true_type"] == pitch_type]
            type_correct = sum(1 for r in type_results if r["correct"])
            type_total = len(type_results)
            type_accuracy = type_correct / type_total * 100 if type_total > 0 else 0
            print(f"{pitch_type}: {type_accuracy:.2f}% ({type_correct}/{type_total})")
    else:
        print("No predictions made.")

if __name__ == "__main__":
    main()