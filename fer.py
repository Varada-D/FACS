from pathlib import Path
import time
import urllib.request

import cv2
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


# ============================================================
# MODEL DOWNLOAD
# ============================================================

def download_model(url: str, destination: Path) -> None:
    """Download model automatically if it does not exist."""

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0:
        print(f"[INFO] Model already exists: {destination}")
        return

    print(f"[INFO] Downloading model: {destination.name}")

    temp_file = destination.with_name(destination.name + ".part")

    try:
        urllib.request.urlretrieve(url, temp_file)

        if temp_file.stat().st_size == 0:
            raise RuntimeError("Downloaded model is empty.")

        temp_file.replace(destination)

        print(f"[INFO] Model downloaded successfully: {destination}")

    except Exception as e:

        if temp_file.exists():
            temp_file.unlink()

        raise RuntimeError(
            f"Failed to download model from {url}: {e}"
        ) from e


# ============================================================
# SOFTMAX
# ============================================================

def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals)


# ============================================================
# DRAW EMOTION BAR
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
        0.45,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )

    bar_x = x + 135

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
        0.45,
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
    face_info: dict[str, str] | None,
    fps: float,
) -> np.ndarray:

    panel = np.full((height, panel_w, 3), 22, dtype=np.uint8)

    x = 15
    y = 28

    cv2.putText(
        panel,
        f"FER+ ONNX (Core Ultra CPU) | FPS: {fps:.1f}",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (120, 220, 255),
        1,
        cv2.LINE_AA,
    )

    y += 28

    if not emotions:

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
        "Emotions",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 210, 90),
        1,
        cv2.LINE_AA,
    )

    y += 22

    for emo_name, score in emotions.items():

        if y + 18 > height:
            break

        draw_bar(
            panel,
            emo_name[:15],
            score,
            x,
            y,
            bar_w=145,
            bar_h=13,
            color=(255, 180, 70),
        )

        y += 20

    y += 14

    # Face information

    cv2.putText(
        panel,
        "Face Crop Info",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (110, 235, 160),
        1,
        cv2.LINE_AA,
    )

    y += 24

    if face_info:

        for key, val in face_info.items():

            if y + 20 > height:
                break

            cv2.putText(
                panel,
                f"{key}: {val}",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )

            y += 22

    return panel


# ============================================================
# FER+ EMOTION PREDICTION
# ============================================================

def predict_emotion(
    net: cv2.dnn.Net,
    gray_frame: np.ndarray,
    box: tuple[int, int, int, int],
) -> dict[str, float] | None:

    x1, y1, x2, y2 = box

    h, w = gray_frame.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

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
# MAIN APPLICATION
# ============================================================

def main(
    camera_index: int = 0,
    frame_skip: int = 2,
    panel_w: int = 360,
    model_path: str = "models/emotion-ferplus-8.onnx",
) -> None:

    if frame_skip < 1:
        raise ValueError("frame_skip must be at least 1.")

    # --------------------------------------------------------
    # Haar Cascade initialization
    # --------------------------------------------------------

    cascade_file = (
        Path(cv2.data.haarcascades)
        / "haarcascade_frontalface_default.xml"
    )

    face_cascade = cv2.CascadeClassifier(str(cascade_file))

    if face_cascade.empty():
        raise RuntimeError(
            f"Could not load Haar cascade from {cascade_file}"
        )

    print("[INFO] Haar Cascade loaded successfully.")

    # --------------------------------------------------------
    # Download FER+ model if missing
    # --------------------------------------------------------

    model_file = Path(model_path)

    download_model(
        FERPLUS_URL,
        model_file,
    )

    # --------------------------------------------------------
    # Load FER+ ONNX model
    # --------------------------------------------------------

    print("[INFO] Loading FER+ ONNX model...")

    net = cv2.dnn.readNetFromONNX(str(model_file))

    net.setPreferableBackend(
        cv2.dnn.DNN_BACKEND_OPENCV
    )

    net.setPreferableTarget(
        cv2.dnn.DNN_TARGET_CPU
    )

    print("[INFO] FER+ model loaded successfully.")

    # --------------------------------------------------------
    # Initialize webcam
    # --------------------------------------------------------

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam index {camera_index}"
        )

    frame_idx = 0

    latest_emotions = None
    latest_face_info = None
    latest_box = None

    prev_tick = cv2.getTickCount()
    fps = 0.0

    print("[INFO] Webcam started.")
    print("[INFO] Press 'q' to quit.")

    try:

        while True:

            ret, frame_bgr = cap.read()

            if not ret:
                print("[WARNING] Could not read webcam frame.")
                break

            # ------------------------------------------------
            # FPS calculation
            # ------------------------------------------------

            now = cv2.getTickCount()

            dt = (
                now - prev_tick
            ) / cv2.getTickFrequency()

            prev_tick = now

            if dt > 0:

                inst_fps = 1.0 / dt

                fps = (
                    0.9 * fps + 0.1 * inst_fps
                    if fps > 0
                    else inst_fps
                )

            # ------------------------------------------------
            # Face detection and emotion prediction
            # ------------------------------------------------

            if frame_idx % frame_skip == 0:

                gray = cv2.cvtColor(
                    frame_bgr,
                    cv2.COLOR_BGR2GRAY,
                )

                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.15,
                    minNeighbors=5,
                    minSize=(60, 60),
                )

                if len(faces) > 0:

                    # Select the largest detected face

                    x, y, w, h = max(
                        faces,
                        key=lambda r: r[2] * r[3],
                    )

                    box = (
                        int(x),
                        int(y),
                        int(x + w),
                        int(y + h),
                    )

                    emotions = predict_emotion(
                        net,
                        gray,
                        box,
                    )

                    if emotions is not None:

                        latest_box = box
                        latest_emotions = emotions

                        top_emo, top_score = max(
                            emotions.items(),
                            key=lambda item: item[1],
                        )

                        latest_face_info = {
                            "Top": f"{top_emo} ({top_score:.2f})",
                            "Size": f"{w}x{h} px",
                        }

                    else:

                        latest_box = None
                        latest_emotions = None
                        latest_face_info = None

                else:

                    latest_box = None
                    latest_emotions = None
                    latest_face_info = None

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
                latest_face_info,
                fps,
            )

            combined = np.hstack(
                (camera_view, panel_view)
            )

            cv2.imshow(
                "Core Ultra CPU FER+ Emotions",
                combined,
            )

            frame_idx += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        cap.release()
        cv2.destroyAllWindows()

        print("[INFO] Application closed.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main(
        camera_index=0,
        frame_skip=2,
        panel_w=360,
    )
