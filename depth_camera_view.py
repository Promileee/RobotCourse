import argparse
import sys

import cv2
import numpy as np


def open_capture(index: int) -> cv2.VideoCapture:
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            return cap
    return cv2.VideoCapture()


def ensure_u16(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint16:
        if frame.ndim == 3 and frame.shape[2] == 2:
            return frame.view(np.uint16).reshape(frame.shape[0], frame.shape[1])
        if frame.ndim == 3 and frame.shape[2] == 1:
            return frame[:, :, 0]
        return frame

    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    return gray.astype(np.uint16) * 256


def main() -> int:
    parser = argparse.ArgumentParser(description="Read the second camera and visualize depth.")
    parser.add_argument("--index", type=int, default=1, help="Camera index (default: 1).")
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5000,
        help="Max depth for visualization (0 = auto, default: 5000).",
    )
    parser.add_argument(
        "--fourcc",
        type=str,
        default="Y16 ",
        help="FOURCC to request (default: 'Y16 ').",
    )
    args = parser.parse_args()

    cap = open_capture(args.index)
    if not cap.isOpened():
        print(f"Failed to open camera index {args.index}.", file=sys.stderr)
        return 1

    if args.fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        depth_u16 = ensure_u16(frame)
        if args.max_depth > 0:
            depth_u16 = np.clip(depth_u16, 0, args.max_depth)

        depth_vis = cv2.normalize(depth_u16, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = depth_vis.astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        cv2.imshow("Depth (JET)", depth_color)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
