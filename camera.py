from openni import openni2
import numpy as np
import cv2

mouse_x, mouse_y = -1, -1
current_depth = None


def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    mouse_x, mouse_y = x, y


if __name__ == "__main__":
    openni2.initialize()
    dev = openni2.Device.open_any()
    print(dev.get_device_info())

    depth_stream = dev.create_depth_stream()
    depth_stream.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open RGB camera")
        depth_stream.stop()
        dev.close()
        exit()

    cv2.namedWindow("depth")
    cv2.setMouseCallback("depth", mouse_callback)

    while True:
        ret, rgb_frame = cap.read()
        if not ret or rgb_frame is None:
            print("Cannot read RGB frame")
            break

        frame = depth_stream.read_frame()
        dframe_data = np.array(frame.get_buffer_as_triplet()).reshape([480, 640, 2])
        dpt1 = np.asarray(dframe_data[:, :, 0], dtype="float32")
        dpt2 = np.asarray(dframe_data[:, :, 1], dtype="float32")
        dpt2 *= 255
        dpt = dpt1 + dpt2
        dpt = dpt[:, ::-1]
        current_depth = dpt

        depth_norm = cv2.normalize(dpt, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        if 0 <= mouse_x < 640 and 0 <= mouse_y < 480:
            val = dpt[mouse_y, mouse_x]
            text = f"({mouse_x}, {mouse_y}), depth={val:.1f}"
            cv2.putText(depth_color, text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.circle(depth_color, (mouse_x, mouse_y), 3, (0, 255, 0), -1)

        cv2.imshow("RGB", rgb_frame)
        cv2.imshow("depth", depth_color)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    depth_stream.stop()
    dev.close()
    cv2.destroyAllWindows()
