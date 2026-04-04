import sys
import types


def install_dependency_stubs():
    if "dotenv" not in sys.modules:
        dotenv = types.ModuleType("dotenv")
        dotenv.load_dotenv = lambda *args, **kwargs: False
        sys.modules["dotenv"] = dotenv

    if "cv2" not in sys.modules:
        cv2 = types.ModuleType("cv2")
        cv2.CAP_PROP_FPS = 5
        cv2.COLOR_BGR2RGB = 1
        cv2.COLOR_BGR2GRAY = 2
        cv2.FONT_HERSHEY_SIMPLEX = 0
        cv2.VideoCapture = object
        cv2.cvtColor = lambda image, code: image
        cv2.resize = lambda image, size: image
        cv2.putText = lambda *args, **kwargs: None
        cv2.imshow = lambda *args, **kwargs: None
        cv2.waitKey = lambda *args, **kwargs: -1
        cv2.destroyAllWindows = lambda: None
        cv2.GaussianBlur = lambda image, kernel, sigma: image
        cv2.absdiff = lambda lhs, rhs: lhs
        cv2.threshold = lambda image, thresh, maxval, mode: (None, image)
        cv2.dilate = lambda image, kernel, iterations=1: image
        cv2.countNonZero = lambda image: 0
        cv2.accumulateWeighted = lambda src, dst, alpha: None
        cv2.convertScaleAbs = lambda src: src
        sys.modules["cv2"] = cv2

    if "firebase_admin" not in sys.modules:
        firebase_admin = types.ModuleType("firebase_admin")
        firebase_admin._apps = []
        firebase_admin.initialize_app = lambda cred: object()
        firebase_admin.credentials = types.SimpleNamespace(
            Certificate=lambda path: path
        )
        firebase_admin.messaging = types.SimpleNamespace(
            Message=object,
            Notification=object,
            AndroidConfig=object,
            AndroidNotification=object,
            APNSConfig=object,
            APNSPayload=object,
            Aps=object,
            send=lambda *args, **kwargs: "sent",
        )
        sys.modules["firebase_admin"] = firebase_admin
