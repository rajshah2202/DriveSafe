from scipy.spatial import distance as dist
from imutils import face_utils

# define two constants, one for the mouth aspect ratio to indicate
# yawn and then a second constant for the number of consecutive
# frames the mouth must be over the threshold for to set off the
# alarm
MOUTH_AR_THRESH = 0.15
MOUTH_AR_CONSEC_FRAMES = 24


def mouth_aspect_ratio(mouth):
    # compute the euclidean distances between the two sets of
    # vertical mouth landmarks (x, y)-coordinates
    A = dist.euclidean(mouth[1], mouth[7])
    B = dist.euclidean(mouth[2], mouth[6])
    C = dist.euclidean(mouth[3], mouth[5])
    # compute the euclidean distance between the horizontal
    # mouth landmark (x, y)-coordinates
    D = dist.euclidean(mouth[0], mouth[4])
    # compute the mouth aspect ratio
    mar = (A + B + C) / (3.0 * D)
    # return the mouth aspect ratio
    return mar


def detect_yawn(predictor, rect, gray, mStart, mEnd, YAWN_COUNTER):
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    # extract the mouth coordinates, then use the
    # coordinates to compute the mouth aspect ratio
    mouth = shape[mStart:mEnd]
    mar = mouth_aspect_ratio(mouth)

    # check to see if the mouth aspect ratio is over the yawn
    # threshold, and if so, increment the yawn frame counter
    if mar > MOUTH_AR_THRESH:
        YAWN_COUNTER += 1
        if YAWN_COUNTER >= MOUTH_AR_CONSEC_FRAMES:
            return True, mouth, YAWN_COUNTER
        return False, mouth, YAWN_COUNTER
    # otherwise, the mouth aspect ratio is not over the yawn
    # threshold, so reset the counter and alarm
    else:
        YAWN_COUNTER = 0
        return False, mouth, YAWN_COUNTER
