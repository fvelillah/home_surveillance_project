import os
import sys
import cv2
import numpy as np


def main():
    argv = sys.argv
    user = os.getenv("DAHUA_NVR_USER", "admin")
    password = os.getenv("DAHUA_NVR_PASSWORD", "your_password")
    ip = os.getenv("DAHUA_NVR_IP", "192.168.1.100")
    port = os.getenv("DAHUA_NVR_HTTP_PORT", "80")
    channel = int(argv[1]) if len(argv) > 1 else 1
    subtype = int(argv[2]) if len(argv) > 2 else 1  # 0 main 1 sub

    url=f"http://{user}:{password}@{ip}:{port}/cgi-bin/mjpg/video.cgi?channel={channel}&subtype={subtype}"

    cap = cv2.VideoCapture(url)

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            print("Error")
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()