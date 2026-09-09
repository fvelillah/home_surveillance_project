import cv2
import numpy as np
import sys


def main():
    argv = sys.argv
    user="ai_surveillance"
    password="fer166326"
    ip="192.168.1.126"
    port="80"
    channel=int(argv[1])
    subtype=int(argv[2]) # 0 main 1 sub

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