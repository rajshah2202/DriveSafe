# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2

from blink_detection import detect_blink
from yawn_detection import detect_yawn
from attention_detection import detect_attention


def sound_alarm(path):
    # play an alarm sound
    playsound.playsound(path)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", type=str, default="./../data/shape_predictor_68_face_landmarks.dat",
                help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default="./../data/alarm.wav",
                help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
BLINK_COUNTER = 0
YAWN_COUNTER = 0
ATTENTION_COUNTER = 0
ALARM_ON = False

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# grab the indexes of the facial landmarks for the mouth
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)
# loop over frames from the video stream
while True:
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    empty_rect = dlib.rectangles()
    if rects == empty_rect:
        if not ALARM_ON:
            ALARM_ON = True
            # check to see if an alarm file was supplied,
            # and if so, start a thread to have the alarm
            # sound played in the background
            if args["alarm"] != "":
                t = Thread(target=sound_alarm,
                           args=(args["alarm"],))
                t.deamon = True
                t.start()

        # draw an alarm on the frame
        cv2.putText(frame, "PAY ATTENTION!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        ALARM_ON = False

    # loop over the face detections
    for rect in rects:
        blink, leftEye, rightEye, BLINK_COUNTER = detect_blink(
            predictor, rect, gray, lStart, lEnd, rStart, rEnd, BLINK_COUNTER)
        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        yawn, mouth, YAWN_COUNTER = detect_yawn(
            predictor, rect, gray, mStart, mEnd, YAWN_COUNTER)
        # compute the convex hull and visualize the mouth
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        if blink or yawn:
            if not ALARM_ON:
                ALARM_ON = True
                # check to see if an alarm file was supplied,
                # and if so, start a thread to have the alarm
                # sound played in the background
                if args["alarm"] != "":
                    t = Thread(target=sound_alarm,
                               args=(args["alarm"],))
                    t.deamon = True
                    t.start()
            # draw an alarm on the frame
            cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            ALARM_ON = False

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
