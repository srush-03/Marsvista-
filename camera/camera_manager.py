import cv2
import time


class CameraManager:

    def __init__(self,
                 camera_type="oakd",
                 width=1280,
                 height=720,
                 fps=30):

        self.camera_type = camera_type
        self.width = width
        self.height = height
        self.fps = fps

        self.device = None
        self.queue = None

        print(f"[CAMERA] Initializing {camera_type} camera")

        if camera_type == "oakd":
            self._init_oakd()
        else:
            self._init_webcam()


    # --------------------------------------------
    # OAK-D Initialization
    # --------------------------------------------

    def _init_oakd(self):

        try:
            import depthai as dai

            pipeline = dai.Pipeline()

            cam = pipeline.create(dai.node.ColorCamera)

            cam.setResolution(
                dai.ColorCameraProperties.SensorResolution.THE_1080_P
            )

            cam.setFps(self.fps)

            cam.setInterleaved(False)

            cam.setPreviewSize(self.width, self.height)

            xout = pipeline.create(dai.node.XLinkOut)

            xout.setStreamName("rgb")

            cam.preview.link(xout.input)

            device = dai.Device(pipeline)

            queue = device.getOutputQueue(
                name="rgb",
                maxSize=4,
                blocking=False
            )

            self.device = device
            self.queue = queue

            print("[CAMERA] OAK-D Lite started")

        except Exception as e:

            print(f"[CAMERA] OAK-D failed: {e}")

            print("[CAMERA] Falling back to webcam")

            self.camera_type = "webcam"

            self._init_webcam()


    # --------------------------------------------
    # Webcam Initialization
    # --------------------------------------------

    def _init_webcam(self):

        for i in range(5):

            cap = cv2.VideoCapture(i)

            if cap.isOpened():

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)

                self.device = cap

                print(f"[CAMERA] Webcam opened at index {i}")

                return

        raise RuntimeError("No camera detected")


    # --------------------------------------------
    # Frame Capture
    # --------------------------------------------

    def get_frame(self):

        if self.camera_type == "oakd":

            packet = self.queue.get()

            frame = packet.getCvFrame()

            return True, frame

        else:

            return self.device.read()


    # --------------------------------------------
    # Release Camera
    # --------------------------------------------

    def release(self):

        print("[CAMERA] Releasing camera")

        if self.camera_type == "webcam":

            self.device.release()

        else:

            self.device.close()