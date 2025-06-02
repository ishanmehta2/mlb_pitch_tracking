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
            'horizontal_movement': abs(x[-1] - x[0]),
            'avg_velocity': np.mean(velocities),
            'max_velocity': np.max(velocities),
            'curvature': curvature,
            'curve_ratio': curve_ratio,
            'total_travel': total_distance
        }
        
        # Print features for debugging
        print(f"Features: curvature={features['curvature']:.6f}, horiz_move={features['horizontal_movement']:.1f}, vert_drop={features['vertical_drop']:.1f}, curve_ratio={features['curve_ratio']:.3f}")
        
        return features
    
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def predict_pitch_type(features):
    """Predict detailed pitch type based on extracted features"""
    if not features:
        return "unknown"
    
    # Extract key metrics
    curvature = features['curvature']
    horizontal_movement = features['horizontal_movement']
    vertical_drop = features['vertical_drop']
    curve_ratio = features['curve_ratio']
    avg_velocity = features['avg_velocity']
    
    # Multi-class classification logic
    # Curveball: high curvature and significant vertical drop
    if curvature > 0.005 and vertical_drop > 80:
        pitch_type = "curveball"
    
    # Slider: moderate to high curvature with significant horizontal movement
    elif curvature > 0.002 and horizontal_movement > 70:
        pitch_type = "slider"
    
    # Changeup: less velocity, moderate movement
    elif avg_velocity < 12 and curve_ratio < 1.15:
        pitch_type = "changeup"
    
    # Sinker: moderate downward and horizontal movement
    elif vertical_drop > 60 and horizontal_movement > 40 and curvature < 0.003:
        pitch_type = "sinker"
    
    # Knucklecurve: specific combination of movement patterns
    elif curvature > 0.004 and vertical_drop > 60 and horizontal_movement > 50:
        pitch_type = "knucklecurve"
    
    # Fastball: high velocity, relatively straight path
    else:
        pitch_type = "fastball"
    
    print(f"Predicted detailed pitch type: {pitch_type}")
    return pitch_type

def map_true_pitch_to_binary(pitch_type):
    """Map true pitch type to binary class
    For ground truth: fastball and sinker are 'fastball', all others are 'offspeed'
    """
    if pitch_type.lower() in ["fastball", "sinker"]:
        return "fastball"
    else:
        return "offspeed"

def map_predicted_pitch_to_binary(pitch_type):
    """Map predicted pitch type to binary class
    For predictions: only fastball is 'fastball', all others (including sinker) are 'offspeed'
    """
    if pitch_type.lower() == "fastball":
        return "fastball"
    else:
        return "offspeed"

def analyze_pitch(video_path):
    """Analyze a pitch video and return detailed and binary classifications"""
    print(f"\nAnalyzing video: {os.path.basename(video_path)}")
    
    # Track the ball
    ball_positions = process_video(video_path)
    
    if ball_positions is not None and len(ball_positions) > 0:
        print(f"Detected {len(ball_positions)} ball positions")
        
        # Extract features
        features = extract_features(ball_positions)
        
        if features:
            # Predict detailed pitch type
            detailed_type = predict_pitch_type(features)
            
            # Map to binary class (for predictions)
            binary_class = map_predicted_pitch_to_binary(detailed_type)
            
            return detailed_type, binary_class
    else:
        print("Ball tracking failed - insufficient ball positions detected")
    
    return "unknown", "unknown"

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
    """Main function to process videos and evaluate classification accuracy"""
    # Define paths
    data_dir = "../baseline_data"
    json_path = "../data/mlb-youtube-segmented.json"
    
    # Load pitch data
    print(f"Loading pitch data from {json_path}")
    pitch_data = load_pitch_data(json_path)
    
    if not pitch_data:
        print("No pitch data found. Exiting.")
        return
    
    print(f"Loaded data for {len(pitch_data)} pitches")
    
    # Get list of video files
    video_files = [f for f in os.listdir(data_dir) if f.endswith(('.mp4', '.mov'))]
    
    if not video_files:
        print(f"No video files found in {data_dir}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    detailed_results = []
    binary_results = []
    detailed_correct = 0
    binary_correct = 0
    total_predictions = 0
    
    # Process each video
    for filename in video_files:
        video_path = os.path.join(data_dir, filename)
        
        # Extract video ID from filename
        video_id = extract_video_id_from_filename(filename)
        
        # Get ground truth type from JSON
        true_detailed_type = "unknown"
        if video_id in pitch_data:
            if "type" in pitch_data[video_id]:
                true_detailed_type = pitch_data[video_id]["type"]
        
        if true_detailed_type == "unknown":
            print(f"Skipping {filename} - no ground truth type found in JSON")
            continue
        
        # Convert specific pitch type to binary class (using different mapping for ground truth)
        true_binary_class = map_true_pitch_to_binary(true_detailed_type)
        
        # Predict pitch types
        predicted_detailed_type, predicted_binary_class = analyze_pitch(video_path)
        
        # Calculate accuracy
        detailed_is_correct = predicted_detailed_type == true_detailed_type
        binary_is_correct = predicted_binary_class == true_binary_class
        
        if detailed_is_correct:
            detailed_correct += 1
        if binary_is_correct:
            binary_correct += 1
            
        total_predictions += 1
        
        # Store results
        detailed_results.append({
            "filename": filename,
            "true_type": true_detailed_type,
            "predicted_type": predicted_detailed_type,
            "correct": detailed_is_correct
        })
        
        binary_results.append({
            "filename": filename,
            "true_type": true_detailed_type,
            "true_binary": true_binary_class,
            "predicted_binary": predicted_binary_class,
            "correct": binary_is_correct
        })
        
        # Print individual result
        print(f"Result: {filename}")
        print(f"  Detailed: True={true_detailed_type}, Predicted={predicted_detailed_type}, Correct={detailed_is_correct}")
        print(f"  Binary: True={true_binary_class}, Predicted={predicted_binary_class}, Correct={binary_is_correct}")
        print('='*80)
    
    # Calculate and print overall accuracy
    if total_predictions > 0:
        detailed_accuracy = detailed_correct / total_predictions * 100
        binary_accuracy = binary_correct / total_predictions * 100
        
        print(f"\nDetailed Classification Accuracy: {detailed_accuracy:.2f}% ({detailed_correct}/{total_predictions})")
        print(f"Binary Classification Accuracy: {binary_accuracy:.2f}% ({binary_correct}/{total_predictions})")
        
        # Get distribution of true and predicted types
        true_detailed = [r["true_type"] for r in detailed_results]
        pred_detailed = [r["predicted_type"] for r in detailed_results]
        
        print("\nTrue Detailed Type Distribution:")
        for pitch_type, count in Counter(true_detailed).items():
            print(f"{pitch_type}: {count} ({count/len(true_detailed)*100:.1f}%)")
        
        print("\nPredicted Detailed Type Distribution:")
        for pitch_type, count in Counter(pred_detailed).items():
            print(f"{pitch_type}: {count} ({count/len(pred_detailed)*100:.1f}%)")
        
        # Binary class distribution
        true_binary = [r["true_binary"] for r in binary_results]
        pred_binary = [r["predicted_binary"] for r in binary_results]
        
        print("\nTrue Binary Distribution:")
        for class_name, count in Counter(true_binary).items():
            print(f"{class_name}: {count} ({count/len(true_binary)*100:.1f}%)")
        
        print("\nPredicted Binary Distribution:")
        for class_name, count in Counter(pred_binary).items():
            print(f"{class_name}: {count} ({count/len(pred_binary)*100:.1f}%)")
        
        # Binary class confusion matrix
        confusion = {
            "fastball": {"fastball": 0, "offspeed": 0},
            "offspeed": {"fastball": 0, "offspeed": 0}
        }
        
        for r in binary_results:
            true_class = r["true_binary"]
            pred_class = r["predicted_binary"]
            if true_class in confusion and pred_class in confusion[true_class]:
                confusion[true_class][pred_class] += 1
        
        print("\nBinary Confusion Matrix:")
        print("                 | Predicted |")
        print("                 | Fastball  | Offspeed |")
        print("-----------------|-----------|---------")
        print(f"True Fastball:   |    {confusion['fastball']['fastball']}     |    {confusion['fastball']['offspeed']}    |")
        print(f"True Offspeed:   |    {confusion['offspeed']['fastball']}     |    {confusion['offspeed']['offspeed']}    |")
        
        # Calculate precision, recall, F1 score for binary classification
        true_pos = confusion['fastball']['fastball']
        false_pos = confusion['offspeed']['fastball']
        true_neg = confusion['offspeed']['offspeed']
        false_neg = confusion['fastball']['offspeed']
        
        precision_fastball = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall_fastball = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1_fastball = 2 * (precision_fastball * recall_fastball) / (precision_fastball + recall_fastball) if (precision_fastball + recall_fastball) > 0 else 0
        
        precision_offspeed = true_neg / (true_neg + false_neg) if (true_neg + false_neg) > 0 else 0
        recall_offspeed = true_neg / (true_neg + false_pos) if (true_neg + false_pos) > 0 else 0
        f1_offspeed = 2 * (precision_offspeed * recall_offspeed) / (precision_offspeed + recall_offspeed) if (precision_offspeed + recall_offspeed) > 0 else 0
        
        print("\nBinary Classification Metrics:")
        print(f"Fastball - Precision: {precision_fastball:.2f}, Recall: {recall_fastball:.2f}, F1: {f1_fastball:.2f}")
        print(f"Offspeed - Precision: {precision_offspeed:.2f}, Recall: {recall_offspeed:.2f}, F1: {f1_offspeed:.2f}")
        
        # List incorrect binary classifications
        print("\nIncorrect Binary Classifications:")
        incorrect = [r for r in binary_results if not r["correct"]]
        for r in incorrect:
            print(f"{r['filename']}: True={r['true_type']} ({r['true_binary']}), Predicted={r['predicted_binary']}")
    else:
        print("No predictions made.")

if __name__ == "__main__":
    main()