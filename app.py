import streamlit as st
import cv2
import os
import numpy as np
import tempfile
import mediapipe as mp
import math
import webbrowser

UPLOAD_FOLDER = 'uploads'
DIFFERENCE_VIDEO_PATH = 'static/difference.mp4'

# Create the uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Streamlit app for video similarity and pose estimation
def main():
    st.title('Video Processing App')

    # Select the processing option (video similarity or pose estimation)
    processing_option = st.selectbox('Select Processing Option:', ['Video Similarity', 'Pose Estimation'])

    if processing_option == 'Video Similarity':
        video_similarity_app()
    elif processing_option == 'Pose Estimation':
        pose_estimation_app()

def video_similarity_app():
    st.subheader('Video Similarity')

    # Upload two videos
    file1 = st.file_uploader('Upload first video (.mp4)', type=['mp4'])
    file2 = st.file_uploader('Upload second video (.mp4)', type=['mp4'])

    if file1 and file2:
        # Process the videos
        process_button = st.button('Process Videos')
        if process_button:
            process_videos(file1, file2)
            st.success('Videos processed successfully!')

            # Display the processed video
            st.video(DIFFERENCE_VIDEO_PATH, format="video/mp4")

def pose_estimation_app():
    st.subheader('Pose Estimation')

    # Upload a video file
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi"])

    if video_file:
        # Convert the uploaded file to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(video_file.read())
        temp_file.close()
        video_path = temp_file.name

        # Display uploaded video
        st.video(video_path)

        # Initialize MediaPipe Pose solution
        mp_pose_holistic = mp.solutions.holistic

        def calculate_angle(a, b, c):
            # Calculate the angle between three points (in degrees)
            angle_radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
            angle_degrees = math.degrees(angle_radians)
            return angle_degrees

        def process_frame(frame):
            # Load the Holistic model from MediaPipe
            with mp_pose_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                # Convert frame to RGB format (required by MediaPipe)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Perform Pose estimation on the frame
                results = holistic.process(frame)

                # Convert frame back to RGB format for Streamlit display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Draw landmarks and calculate angles
                if results.pose_landmarks:
                    # Extract key points
                    landmarks = results.pose_landmarks.landmark

                    # Calculate angles between body parts
                    left_shoulder = landmarks[mp_pose_holistic.PoseLandmark.LEFT_SHOULDER]
                    left_elbow = landmarks[mp_pose_holistic.PoseLandmark.LEFT_ELBOW]
                    left_wrist = landmarks[mp_pose_holistic.PoseLandmark.LEFT_WRIST]

                    right_shoulder = landmarks[mp_pose_holistic.PoseLandmark.RIGHT_SHOULDER]
                    right_elbow = landmarks[mp_pose_holistic.PoseLandmark.RIGHT_ELBOW]
                    right_wrist = landmarks[mp_pose_holistic.PoseLandmark.RIGHT_WRIST]

                    neck = landmarks[mp_pose_holistic.PoseLandmark.NOSE]

                    left_hip = landmarks[mp_pose_holistic.PoseLandmark.LEFT_HIP]
                    left_knee = landmarks[mp_pose_holistic.PoseLandmark.LEFT_KNEE]
                    left_ankle = landmarks[mp_pose_holistic.PoseLandmark.LEFT_ANKLE]

                    right_hip = landmarks[mp_pose_holistic.PoseLandmark.RIGHT_HIP]
                    right_knee = landmarks[mp_pose_holistic.PoseLandmark.RIGHT_KNEE]
                    right_ankle = landmarks[mp_pose_holistic.PoseLandmark.RIGHT_ANKLE]

                    # Calculate the angles
                    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                    neck_angle = calculate_angle(left_shoulder, neck, right_shoulder)
                    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

                    # Draw skeletal lines
                    mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks)
                    draw_skeletal_lines(frame, results.pose_landmarks)

                    # Display the angles on the frame
                    cv2.putText(frame, f"Left Elbow Angle: {left_elbow_angle:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, f"Right Elbow Angle: {right_elbow_angle:.2f} degrees", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, f"Neck Angle: {neck_angle:.2f} degrees", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, f"Left Knee Angle: {left_knee_angle:.2f} degrees", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, f"Right Knee Angle: {right_knee_angle:.2f} degrees", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            return frame  # Return the processed frame

        def draw_skeletal_lines(frame, landmarks):
            # Define connections for drawing skeletal lines
            connections = mp_pose_holistic.POSE_CONNECTIONS

            # Loop through the connections and draw lines
            for connection in connections:
                start_point = connection[0]
                end_point = connection[1]

                # Get the landmark points
                start_landmark = landmarks.landmark[start_point]
                end_landmark = landmarks.landmark[end_point]

                # Convert landmark positions to pixel coordinates
                x1, y1 = int(start_landmark.x * frame.shape[1]), int(start_landmark.y * frame.shape[0])
                x2, y2 = int(end_landmark.x * frame.shape[1]), int(end_landmark.y * frame.shape[0])

                # Draw the skeletal lines on the frame
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Process the video and apply pose estimation
        cap = cv2.VideoCapture(video_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame
            processed_frame = process_frame(frame)

            # Display the processed frame
            st.image(processed_frame, channels="RGB", use_column_width=True)

if __name__ == '__main__':
    main()
