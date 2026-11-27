from pathlib import Path
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

EMOTION_LABELS = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
]

FERPLUS_URL = (
    "https://github.com/onnx/models/raw/main/"
    "validated/vision/body_analysis/emotion_ferplus/model/"
    "emotion-ferplus-8.onnx"
)

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/"
    "face_landmarker.task"
)


# ============================================================
# MODEL DOWNLOAD
# ============================================================

def download_model(url: str, destination: Path) -> None:
    """Download a model if it does not already exist."""

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0:
        print(f"[INFO] Model already exists: {destination}")
        return

    print(f"[INFO] Downloading: {destination.name}")

    temp_file = destination.with_name(destination.name + ".part")

    try:
        urllib.request.urlretrieve(url, temp_file)

        if temp_file.stat().st_size == 0:
            raise RuntimeError("Downloaded model is empty.")

        temp_file.replace(destination)

        print(f"[INFO] Download completed: {destination}")

    except Exception as e:
        if temp_file.exists():
            temp_file.unlink()

        raise RuntimeError(
            f"Failed to download {destination.name}: {e}"
        ) from e


# ============================================================
# SOFTMAX
# ============================================================

def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals)


# ============================================================
# DRAW EMOTION / BLENDSHAPE BAR
# ============================================================

def draw_bar(
    img: np.ndarray,
    label: str,
    value: float,
    x: int,
    y: int,
    bar_w: int = 160,
    bar_h: int = 14,
    color: tuple[int, int, int] = (80, 200, 120),
) -> None:

    clamped = float(np.clip(value, 0.0, 1.0))
    fill_w = int(round(clamped * bar_w))

    cv2.putText(
        img,
        label,
        (x, y + bar_h - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )

    bar_x = x + 145

    cv2.rectangle(
        img,
        (bar_x, y),
        (bar_x + bar_w, y + bar_h),
        (55, 55, 55),
        -1,
    )

    cv2.rectangle(
        img,
        (bar_x, y),
        (bar_x + fill_w, y + bar_h),
        color,
        -1,
    )

    cv2.rectangle(
        img,
        (bar_x, y),
        (bar_x + bar_w, y + bar_h),
        (130, 130, 130),
        1,
    )

    cv2.putText(
        img,
        f"{clamped:.2f}",
        (bar_x + bar_w + 8, y + bar_h - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )


# ============================================================
# SIDE PANEL
# ============================================================

def build_side_panel(
    height: int,
    panel_w: int,
    emotions: dict[str, float] | None,
    blendshapes: dict[str, float] | None,
    fps: float,
) -> np.ndarray:

    panel = np.full((height, panel_w, 3), 22, dtype=np.uint8)

    x = 15
    y = 28

    cv2.putText(
        panel,
        f"MediaPipe + FER+ | FPS: {fps:.1f}",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (120, 220, 255),
        1,
        cv2.LINE_AA,
    )

    y += 28

    if not emotions and not blendshapes:

        cv2.putText(
            panel,
            "No face detected",
            (x, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        return panel

    # Emotion scores

    cv2.putText(
        panel,
        "Emotions (FER+)",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 210, 90),
        1,
        cv2.LINE_AA,
    )

    y += 22

    if emotions:

        for emo_name, score in emotions.items():

            if y + 18 > height:
                break

            draw_bar(
                panel,
                emo_name[:16],
                score,
                x,
                y,
                bar_w=135,
                bar_h=13,
                color=(255, 180, 70),
            )

            y += 20

    y += 12

    # Blendshape scores

    cv2.putText(
        panel,
        "Top Blendshapes",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (110, 235, 160),
        1,
        cv2.LINE_AA,
    )

    y += 22

    if blendshapes:

        top_bs = sorted(
            blendshapes.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:8]

        for bs_name, score in top_bs:

            if y + 18 > height:
                break

            draw_bar(
                panel,
                bs_name[:17],
                score,
                x,
                y,
                bar_w=135,
                bar_h=13,
                color=(90, 215, 130),
            )

            y += 20

    return panel


# ============================================================
# FACE LANDMARKS TO BOUNDING BOX
# ============================================================

def landmarks_to_box(
    landmarks: list,
    frame_w: int,
    frame_h: int,
    padding_ratio: float = 0.08,
) -> tuple[int, int, int, int] | None:

    if not landmarks:
        return None

    xs = [lm.x * frame_w for lm in landmarks]
    ys = [lm.y * frame_h for lm in landmarks]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    box_w = max_x - min_x
    box_h = max_y - min_y

    pad_x = box_w * padding_ratio
    pad_y = box_h * padding_ratio

    x1 = max(0, int(min_x - pad_x))
    y1 = max(0, int(min_y - pad_y))

    x2 = min(frame_w, int(max_x + pad_x))
    y2 = min(frame_h, int(max_y + pad_y))

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)


# ============================================================
# FER+ EMOTION PREDICTION
# ============================================================

def predict_emotion(
    net: cv2.dnn.Net,
    gray_frame: np.ndarray,
    box: tuple[int, int, int, int],
) -> dict[str, float] | None:

    x1, y1, x2, y2 = box

    face_gray = gray_frame[y1:y2, x1:x2]

    if face_gray.size == 0:
        return None

    resized = cv2.resize(
        face_gray,
        (64, 64),
        interpolation=cv2.INTER_AREA,
    )

    blob = resized.astype(np.float32).reshape(1, 1, 64, 64)

    net.setInput(blob)

    logits = net.forward().flatten()

    if logits.size != len(EMOTION_LABELS):
        raise ValueError(
            f"Expected 8 FER+ outputs, received {logits.size}."
        )

    probs = softmax(logits)

    return {
        label: float(prob)
        for label, prob in zip(EMOTION_LABELS, probs)
    }


# ============================================================
# MAIN
# ============================================================

def main(
    camera_index: int = 0,
    frame_skip: int = 2,
    panel_w: int = 380,
    ferplus_path: str = "models/emotion-ferplus-8.onnx",
    landmarker_path: str = "models/face_landmarker.task",
) -> None:

    # --------------------------------------------------------
    # Download models automatically
    # --------------------------------------------------------

    ferplus_file = Path(ferplus_path)
    landmarker_file = Path(landmarker_path)

    download_model(
        FERPLUS_URL,
        ferplus_file,
    )

    download_model(
        FACE_LANDMARKER_URL,
        landmarker_file,
    )

    # --------------------------------------------------------
    # Load FER+ ONNX model
    # --------------------------------------------------------

    print("[INFO] Loading FER+ model...")

    net = cv2.dnn.readNetFromONNX(str(ferplus_file))

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # --------------------------------------------------------
    # Load MediaPipe FaceLandmarker
    # --------------------------------------------------------

    print("[INFO] Loading MediaPipe FaceLandmarker...")

    base_options = mp_python.BaseOptions(
        model_asset_path=str(landmarker_file)
    )

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )

    landmarker = vision.FaceLandmarker.create_from_options(options)

    # --------------------------------------------------------
    # Initialize webcam
    # --------------------------------------------------------

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        landmarker.close()
        raise RuntimeError(
            f"Could not open webcam index {camera_index}"
        )

    frame_idx = 0

    latest_emotions = None
    latest_blendshapes = None
    latest_box = None

    prev_tick = cv2.getTickCount()
    fps = 0.0

    start_time = time.monotonic()

    print("[INFO] Webcam started.")
    print("[INFO] Press 'q' to quit.")

    try:

        while True:

            ret, frame_bgr = cap.read()

            if not ret:
                print("[WARNING] Could not read webcam frame.")
                break

            # FPS calculation

            now = cv2.getTickCount()

            dt = (now - prev_tick) / cv2.getTickFrequency()

            prev_tick = now

            if dt > 0:

                inst_fps = 1.0 / dt

                fps = (
                    0.9 * fps + 0.1 * inst_fps
                    if fps > 0
                    else inst_fps
                )

            # ------------------------------------------------
            # Process every Nth frame
            # ------------------------------------------------

            if frame_idx % frame_skip == 0:

                frame_h, frame_w = frame_bgr.shape[:2]

                frame_rgb = cv2.cvtColor(
                    frame_bgr,
                    cv2.COLOR_BGR2RGB,
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame_rgb,
                )

                timestamp_ms = int(
                    (time.monotonic() - start_time) * 1000
                )

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                if result.face_landmarks:

                    face_lms = result.face_landmarks[0]

                    box = landmarks_to_box(
                        face_lms,
                        frame_w,
                        frame_h,
                    )

                    # Blendshape extraction

                    bs_dict = {}

                    if result.face_blendshapes:

                        for cat in result.face_blendshapes[0]:

                            if cat.category_name != "_neutral":

                                bs_dict[cat.category_name] = float(
                                    cat.score
                                )

                    # FER+ prediction

                    gray = cv2.cvtColor(
                        frame_bgr,
                        cv2.COLOR_BGR2GRAY,
                    )

                    emotions = (
                        predict_emotion(net, gray, box)
                        if box is not None
                        else None
                    )

                    latest_box = box

                    latest_blendshapes = (
                        bs_dict if bs_dict else None
                    )

                    latest_emotions = emotions

                else:

                    latest_box = None
                    latest_blendshapes = None
                    latest_emotions = None

            # ------------------------------------------------
            # Draw camera view
            # ------------------------------------------------

            camera_view = frame_bgr.copy()

            if latest_box is not None:

                x1, y1, x2, y2 = latest_box

                cv2.rectangle(
                    camera_view,
                    (x1, y1),
                    (x2, y2),
                    (120, 220, 255),
                    2,
                )

                if latest_emotions:

                    top_emo = max(
                        latest_emotions.items(),
                        key=lambda item: item[1],
                    )[0]

                    cv2.putText(
                        camera_view,
                        top_emo,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (120, 220, 255),
                        2,
                        cv2.LINE_AA,
                    )

            # ------------------------------------------------
            # Build side panel
            # ------------------------------------------------

            h = camera_view.shape[0]

            panel_view = build_side_panel(
                h,
                panel_w,
                latest_emotions,
                latest_blendshapes,
                fps,
            )

            combined = np.hstack(
                (camera_view, panel_view)
            )

            cv2.imshow(
                "MediaPipe Blendshapes + FER+",
                combined,
            )

            frame_idx += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()

        print("[INFO] Application closed.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main(
        camera_index=0,
        frame_skip=2,
        panel_w=380,
    )