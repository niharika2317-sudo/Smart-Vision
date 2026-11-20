import cv2
import sys
import os


def camera_backend_candidates(cv2):
    """Return preferred camera backends for this operating system."""
    if os.name == "nt":
        return [("CAP_DSHOW", cv2.CAP_DSHOW), ("CAP_MSMF", cv2.CAP_MSMF), ("DEFAULT", cv2.CAP_ANY)]
    if sys.platform == "darwin":
        return [("CAP_AVFOUNDATION", getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)),
                ("DEFAULT", cv2.CAP_ANY)]
    return [("DEFAULT", cv2.CAP_ANY)]


def camera_permission_hint():
    if sys.platform == "darwin":
        return ("No camera opened. Allow Terminal or VS Code in System Settings > Privacy & Security > Camera, "
                "then restart the app and try again. Also close apps that may be using the camera.")
    if os.name == "nt":
        return "No camera opened. Close other camera apps and check Windows camera privacy permission."
    return "No camera opened. Check that a camera is connected, not in use, and accessible to this user."


def get_video_source(argument=None):
    argument = argument if argument is not None else (sys.argv[1] if len(sys.argv) > 1 else None)
    if argument is not None:
        if argument.isdigit():
            requested=int(argument)
            indices=[requested]+[i for i in range(4) if i!=requested]
            attempts=[]
            for index in indices:
                for _, backend in camera_backend_candidates(cv2):
                    cap=cv2.VideoCapture(index,backend)
                    if cap.isOpened(): return cap
                    attempts.append(f"index {index}, backend {backend}");cap.release()
            print(camera_permission_hint() + " Tried " + "; ".join(attempts) + ".")
            return cv2.VideoCapture()
        cap = cv2.VideoCapture(os.fspath(argument))
        if not cap.isOpened(): print(f"Could not open video source: {argument}")
        return cap

    print("\n========== VantaWatch ==========")
    print("1. Live Webcam")
    print("2. Video File")
    print("===============================\n")

    choice = input("Choose option (1/2): ")

    if choice == "2":

        path = input("Enter video path: ")

        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): print(f"Could not open video source: {path}")
        return cap

    return get_video_source("0")
