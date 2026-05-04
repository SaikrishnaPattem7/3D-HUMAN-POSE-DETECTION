import os
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Default model path — download pose_landmarker_lite.task from:
# https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
_DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'pose_landmarker_lite.task',
)

# MediaPipe landmark indices → COCO 17-joint indices
# (indices unchanged between old solutions API and new Tasks API)
MP_TO_COCO = {
    0: 0,   # nose
    2: 1,   # left_eye
    5: 2,   # right_eye
    7: 3,   # left_ear
    8: 4,   # right_ear
    11: 5,  # left_shoulder
    12: 6,  # right_shoulder
    13: 7,  # left_elbow
    14: 8,  # right_elbow
    15: 9,  # left_wrist
    16: 10, # right_wrist
    23: 11, # left_hip
    24: 12, # right_hip
    25: 13, # left_knee
    26: 14, # right_knee
    27: 15, # left_ankle
    28: 16, # right_ankle
}

JOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]

NUM_JOINTS = 17


class PoseDetector:
    """
    2D pose detector using MediaPipe Pose Landmarker (Tasks API, v0.10+).
    Outputs 17 keypoints in COCO format as normalized (x, y) coordinates.
    """

    def __init__(
        self,
        model_path=None,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        model_path = model_path or _DEFAULT_MODEL
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"MediaPipe model not found: {model_path}\n"
                "Download it with:\n"
                "  curl -L -o pose_landmarker_lite.task \\\n"
                "    https://storage.googleapis.com/mediapipe-models/"
                "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
            )

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def detect(self, image):
        """
        Detect 2D keypoints from a single BGR image.

        Args:
            image: np.array (H, W, 3) in BGR format

        Returns:
            keypoints: np.array (17, 2) with normalized (x, y) in [0, 1],
                       or None if no pose is detected
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return None

        landmarks = result.pose_landmarks[0]
        keypoints = np.zeros((NUM_JOINTS, 2), dtype=np.float32)

        for mp_idx, coco_idx in MP_TO_COCO.items():
            lm = landmarks[mp_idx]
            keypoints[coco_idx] = [lm.x, lm.y]

        return keypoints

    def detect_video(self, video_path):
        """
        Detect 2D keypoints for every frame of a video.

        Args:
            video_path: path to video file

        Returns:
            frames_keypoints: list of np.array (17, 2) per frame;
                              entry is None if no pose was detected in that frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        frames_keypoints = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames_keypoints.append(self.detect(frame))

        cap.release()
        return frames_keypoints

    def draw_keypoints(self, image, keypoints):
        """Overlay detected keypoints on a copy of the image."""
        if keypoints is None:
            return image
        h, w = image.shape[:2]
        vis = image.copy()
        for x, y in keypoints:
            cv2.circle(vis, (int(x * w), int(y * h)), 5, (0, 255, 0), -1)
        return vis

    def close(self):
        self._landmarker.close()
