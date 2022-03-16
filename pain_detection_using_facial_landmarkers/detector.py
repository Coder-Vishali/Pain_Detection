# import the necessary packages
from imutils import face_utils
import dlib
import cv2
import os
from scipy.spatial import distance as dist
import pandas as pd
import csv
import shutil
import time

def compute_brow_lowering(shape):
    """
        Function to compute effective brow lowering value.
        The distance of inner eyebrows divided by the distance of
        outer eyebrows, the quadratic polynomial coefficients of the right and the left eyebrows.

        Args:
            shape(numpy): Numpy array 68 points facial landmarks

        Returns:
            float: This function returns the effective brow lowering value
    """
    print("\nComputing brow lowering value (AU 4) ")
    inner_eyebrow_dist = dist.euclidean(shape[21], shape[22])
    print(f"Inner eyebrow distance: {inner_eyebrow_dist}")
    outer_eyebrow_dist = dist.euclidean(shape[17], shape[26])
    print(f"Outer eyebrow distance: {outer_eyebrow_dist}")
    brow_lowering = inner_eyebrow_dist/outer_eyebrow_dist
    print(f"Brow lowering value: {brow_lowering}")
    if brow_lowering < 0.13:
        brow_lowering_status = 1
    else:
        brow_lowering_status = 0
    return brow_lowering, brow_lowering_status


def wrinkling_of_nose(shape):
    """
        Function to compute effective wrinkling of nose value.
        The normalized distance between nose and philtrum and of nasolabial folds.

        Args:
            shape(numpy): Numpy array 68 points facial landmarks

        Returns:
            float: This function returns the effective wrinkling of nose value
    """
    print("\nComputing wrinkling of nose (AU 9) ")
    philtrum_dist = dist.euclidean(shape[30], shape[33])
    print(f"Philtrum distance: {philtrum_dist}")
    nasolabial_folds_dist = dist.euclidean(shape[31], shape[35])
    print(f"Nasolabial folds distance: {nasolabial_folds_dist}")
    wrinkling_nose = abs(philtrum_dist * nasolabial_folds_dist)
    print(f"Wrinkling nose value: {wrinkling_nose}")
    if wrinkling_nose < 340:
        wrinkling_nose_status = 1
    else:
        wrinkling_nose_status = 0
    return wrinkling_nose, wrinkling_nose_status


def tightening_of_eyelids(shape):
    """
        Function to compute effective tightening of eyelids value.
        The distance between the inner eye corners divided by the distance of outer eye corners.

        Args:
            shape(numpy): Numpy array 68 points facial landmarks

        Returns:
            float: This function returns the tightening of eyelids
    """
    print("\nComputing tightening of eyelids (AU 7) ")
    inner_eye_dist = dist.euclidean(shape[39], shape[42])
    print(f"Inner eye corner distance: {inner_eye_dist}")
    outer_eye_dist = dist.euclidean(shape[36], shape[45])
    print(f"Outer eye corner distance: {outer_eye_dist}")
    tightening_eyelids = inner_eye_dist/outer_eye_dist
    print(f"Tightening of eyes value: {tightening_eyelids}")
    if tightening_eyelids > 0.41:
        tightening_eye_status = 1
    else:
        tightening_eye_status = 0
    return tightening_eyelids, tightening_eye_status


def closing_of_eyes(shape):
    """
        Function to compute effective closing eyes value.
        The distance of upper and lower eyelids divided by the distance from the head to the corner of the eyes.

        Args:
            shape(numpy): Numpy array 68 points facial landmarks

        Returns:
            float: This function returns the closing eyes value.
    """
    print("\nComputing closing of eyes (AU 43) ")
    eyelid_dist = dist.euclidean(shape[44], shape[46])
    print(f"Eyelid distance: {eyelid_dist}")
    eye_dist = dist.euclidean(shape[42], shape[45])
    print(f"Eye corner distance: {eye_dist}")
    closing_eye = eyelid_dist/eye_dist
    print(f"Closing of eyes value: {closing_eye}")
    if closing_eye < 0.16:
        closing_eye_status = 1
    else:
        closing_eye_status = 0
    return closing_eye, closing_eye_status


def capture_video():
    """
        Function to capture the live video feed and analyze the pain detection of the person.
    """
    # define a video capture object
    vid = cv2.VideoCapture(0)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = cv2.VideoWriter(os.path.join(output_path, 'pain_video.mp4'),
                                   cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height))

    start_time = time.time()
    current_time = time.time()
    # while True:
    while current_time - start_time < 60:
        current_time = time.time()
        ret, frame = vid.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        # detect faces in the grayscale image

        # loop over the face detections
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Compute brow lowering value (AU 4)
            au4, au4_status = compute_brow_lowering(shape)

            # Compute wrinkling of nose value (AU 9)
            au9, au9_status = wrinkling_of_nose(shape)

            # Compute tightening of eyelids value (AU 7)
            au7, au7_status = tightening_of_eyelids(shape)

            # Compute closing of eyes value (AU 43)
            au43, au43_status = closing_of_eyes(shape)

            pain = au4 + au9 + au7 + au43
            pain = round(pain, 2)
            print(f"\nPain value: {pain}")


            if au4_status and au7_status and au43_status and au7_status:
                pain_text = "Pain"
            else:
                pain_text = "No Pain"

            # Dump values into csv file
            with open(pain_file, 'a') as file_pain:
                pain_writer = csv.writer(file_pain)
                pain_writer.writerow([au4, au9, au7, au43, pain, pain_text])
            
            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.putText(frame, "Pain value : {}".format(pain), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Pain Text : {}".format(pain_text), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Brow lowering: {}".format(au4_status), (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Wrinkling of nose: {}".format(au9_status), (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Tightening of eyelids: {}".format(au7_status), (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Closing of eye: {}".format(au43_status), (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # show the face number
            cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        video_writer.write(frame)
        cv2.imshow('Video', frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            vid.release()
            video_writer.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(os.path.join(os.getcwd(), r"Lib/shape_predictor_68_face_landmarks.dat"))

    # Output file path
    output_path = os.path.join(os.getcwd(), 'Output')
    # Create a output directory if its not available
    try:
        os.mkdir(output_path)
    except OSError as error:
        print('Warning : Output directory already exists, running the script might overwrite the existing files !')

    # Define the csv file to write data
    pain_file = os.path.join(output_path, 'pain.csv')
    with open(pain_file, 'w') as file_pain:
        writer = csv.writer(file_pain)
        writer.writerow(['brow_lowering', 'wrinkling_nose', 'tightening_eyelids', 'closing_eyes', 'pain', 'label'])

    # start the video capture
    capture_video()

    # Analysis the data
    df = pd.read_csv(os.path.join(output_path, 'pain.csv'))
    print("\nDescription of the pain csv file:\n", df.describe())
    df.describe()

    # To zip the output file
    shutil.make_archive('Output', 'zip', os.path.join(os.getcwd(), 'Output'))
