import os
import tempfile
import cv2
import numpy as np
import pandas as pd
import torch
from feat import Detectorv2


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
    """Draw a label, normalized [0, 1] bar, and numeric score onto img."""
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
    cv2.rectangle(img, (bar_x, y), (bar_x + bar_w, y + bar_h), (55, 55, 55), -1)
    cv2.rectangle(img, (bar_x, y), (bar_x + fill_w, y + bar_h), color, -1)
    cv2.rectangle(img, (bar_x, y), (bar_x + bar_w, y + bar_h), (130, 130, 130), 1)

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


def build_side_panel(
    height: int,
    panel_w: int,
    emotions: dict[str, float] | None,
    aus: dict[str, float] | None,
    fps: float,
) -> np.ndarray:
    """Create a dedicated side panel beside the camera feed."""
    panel = np.full((height, panel_w, 3), 22, dtype=np.uint8)

    x = 15
    y = 28

    cv2.putText(
        panel,
        f"Py-Feat Live | Display FPS: {fps:.1f}",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (120, 220, 255),
        1,
        cv2.LINE_AA,
    )
    y += 28

    if not emotions and not aus:
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

    if emotions:
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

    y += 12

    cv2.putText(
        panel,
        "Top Action Units (Intensity / Score)",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (110, 235, 160),
        1,
        cv2.LINE_AA,
    )
    y += 22

    if aus:
        top_aus = sorted(aus.items(), key=lambda item: item[1], reverse=True)[:8]
        for au_name, intensity in top_aus:
            if y + 18 > height:
                break
            short_name = au_name.split("_")[0] if "_" in au_name else au_name
            label = (
                f"{short_name} ({au_name.split('_', 1)[1][:8]})"
                if "_" in au_name
                else au_name[:15]
            )
            draw_bar(
                panel,
                label,
                intensity,
                x,
                y,
                bar_w=145,
                bar_h=13,
                color=(90, 215, 130),
            )
            y += 20

    return panel


def main(camera_index: int = 0, frame_skip: int = 3, panel_w: int = 360) -> None:
    device = get_device()
    print(f"Loading Detectorv2 on {device}...")
    detector = Detectorv2(device=device)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {camera_index}")

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(tmp_fd)

    frame_idx = 0
    latest_emotions: dict[str, float] | None = None
    latest_aus: dict[str, float] | None = None
    latest_box: tuple[int, int, int, int] | None = None
    prev_tick = cv2.getTickCount()
    fps = 0.0

    print("Press 'q' to quit.")
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            now = cv2.getTickCount()
            dt = (now - prev_tick) / cv2.getTickFrequency()
            prev_tick = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

            if frame_idx % frame_skip == 0:
                cv2.imwrite(tmp_path, frame_bgr)
                fex = detector.detect(tmp_path, data_type="image")

                if fex is not None and len(fex) > 0:
                    row_idx = fex.index[0]
                    latest_emotions = {
                        col: float(val)
                        for col, val in fex.emotions.loc[row_idx].items()
                        if pd.notna(val)
                    }
                    latest_aus = {
                        col: float(val)
                        for col, val in fex.aus.loc[row_idx].items()
                        if pd.notna(val)
                    }

                    fb = fex.facebox
                    if fb is not None and len(fb) > 0:
                        bx = fb.loc[row_idx]
                        if {"FaceRectX", "FaceRectY", "FaceRectWidth", "FaceRectHeight"}.issubset(bx.index):
                            x1 = int(bx["FaceRectX"])
                            y1 = int(bx["FaceRectY"])
                            x2 = int(x1 + bx["FaceRectWidth"])
                            y2 = int(y1 + bx["FaceRectHeight"])
                            latest_box = (x1, y1, x2, y2)
                else:
                    latest_emotions = None
                    latest_aus = None
                    latest_box = None

            camera_view = frame_bgr.copy()
            if latest_box is not None:
                x1, y1, x2, y2 = latest_box
                cv2.rectangle(camera_view, (x1, y1), (x2, y2), (120, 220, 255), 2)

            h, _, _ = camera_view.shape
            panel_view = build_side_panel(h, panel_w, latest_emotions, latest_aus, fps)
            combined = np.hstack((camera_view, panel_view))

            cv2.imshow("Py-Feat Real-Time Emotions & AUs", combined)

            frame_idx += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    main(camera_index=0, frame_skip=3, panel_w=360)