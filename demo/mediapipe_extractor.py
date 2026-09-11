import cv2
import os
import imageio
import numpy as np
import mediapipe as mp

class MediaPipePoseExtractor:
    def __init__(self, static_image_mode=False, model_complexity=1, smooth_landmarks=True):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Neutral lower body template in real meters relative to hip
        self.neutral_lower_body = np.array([
            [-0.12, -0.10,  0.00],  # 12 left hip
            [-0.12, -0.45,  0.00],  # 13 left knee
            [-0.12, -0.80,  0.00],  # 14 left ankle
            [-0.12, -0.90,  0.10],  # 15 left foot
            [ 0.12, -0.10,  0.00],  # 16 right hip
            [ 0.12, -0.45,  0.00],  # 17 right knee
            [ 0.12, -0.80,  0.00],  # 18 right ankle
            [ 0.12, -0.90,  0.10],  # 19 right foot
        ], dtype=np.float32)

    def extract_landmarks(self, frame_bgr):
        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if not results.pose_world_landmarks:
            return None, frame_bgr

        # 3D Metric World Landmarks
        w_lm = results.pose_world_landmarks.landmark
        mp_pts = np.zeros((33, 3), dtype=np.float32)

        for i in range(33):
            mp_pts[i] = [w_lm[i].x, -w_lm[i].y, w_lm[i].z * 0.2]

        ntu_joints = np.zeros((25, 3), dtype=np.float32)

        ntu_joints[0] = (mp_pts[23] + mp_pts[24]) / 2.0
        ntu_joints[20] = (mp_pts[11] + mp_pts[12]) / 2.0
        ntu_joints[1] = (ntu_joints[0] + ntu_joints[20]) / 2.0
        ntu_joints[2] = ntu_joints[20] + np.array([0, 0.10, 0], dtype=np.float32)
        ntu_joints[3] = mp_pts[0]

        ntu_joints[4]  = mp_pts[11] # Left shoulder
        ntu_joints[5]  = mp_pts[13] # Left elbow
        ntu_joints[6]  = mp_pts[15] # Left wrist
        ntu_joints[7]  = mp_pts[19] # Left hand (index)
        ntu_joints[21] = mp_pts[17] # Left hand tip (pinky)
        ntu_joints[22] = mp_pts[21] # Left thumb

        ntu_joints[8]  = mp_pts[12] # Right shoulder
        ntu_joints[9]  = mp_pts[14] # Right elbow
        ntu_joints[10] = mp_pts[16] # Right wrist
        ntu_joints[11] = mp_pts[20] # Right hand (index)
        ntu_joints[23] = mp_pts[18] # Right hand tip (pinky)
        ntu_joints[24] = mp_pts[22] # Right thumb

        hip_vis = w_lm[23].visibility > 0.5 and w_lm[24].visibility > 0.5
        if hip_vis:
            ntu_joints[12] = mp_pts[23]
            ntu_joints[13] = mp_pts[25]
            ntu_joints[14] = mp_pts[27]
            ntu_joints[15] = mp_pts[31]
            ntu_joints[16] = mp_pts[24]
            ntu_joints[17] = mp_pts[26]
            ntu_joints[18] = mp_pts[28]
            ntu_joints[19] = mp_pts[32]
        else:
            base_hip = ntu_joints[0]
            ntu_joints[12:20] = self.neutral_lower_body + base_hip

        annotated_frame = frame_bgr.copy()
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

        return ntu_joints, annotated_frame

    def process_video_file(self, video_path, manual_rotation=None):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        rotate_code = None
        if manual_rotation in (90,):
            rotate_code = cv2.ROTATE_90_CLOCKWISE
        elif manual_rotation in (180,):
            rotate_code = cv2.ROTATE_180
        elif manual_rotation in (270,):
            rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE

        if rotate_code is None:
            try:
                rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
                if rotation == 90:
                    rotate_code = cv2.ROTATE_90_CLOCKWISE
                elif rotation == 180:
                    rotate_code = cv2.ROTATE_180
                elif rotation == 270:
                    rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE
            except Exception:
                rotate_code = None

        skeleton_sequence = []
        annotated_rgb_frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if rotate_code is not None:
                frame = cv2.rotate(frame, rotate_code)

            joints, ann_frame_bgr = self.extract_landmarks(frame)
            if joints is not None:
                skeleton_sequence.append(joints)
            elif len(skeleton_sequence) > 0:
                skeleton_sequence.append(skeleton_sequence[-1])

            # Convert BGR to RGB for imageio video encoding
            ann_frame_rgb = cv2.cvtColor(ann_frame_bgr, cv2.COLOR_BGR2RGB)
            annotated_rgb_frames.append(ann_frame_rgb)

        cap.release()

        if len(skeleton_sequence) == 0:
            return None, None

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_video_path = os.path.join(project_root, "output_annotated.mp4")
        imageio.mimsave(out_video_path, annotated_rgb_frames, fps=fps, codec='libx264', macro_block_size=1)

        return np.array(skeleton_sequence, dtype=np.float32), out_video_path