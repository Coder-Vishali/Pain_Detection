from pain_detector import PainDetector
import cv2
import time
import argparse

parser = argparse.ArgumentParser(description='Trains!')
parser.add_argument('-unbc_only', action='store_true', default=False, help='Load the checkpoint that was only trained on UNBC. Otherwise loaded the checkpoint that was train on Both UNBC and UofR datasets')
parser.add_argument('-test_framerate', action='store_true', default=False, help='Runs frame rate test as well')
args = parser.parse_args()

if args.unbc_only:
    pain_detector = PainDetector(image_size=160, checkpoint_path='checkpoints/59448122/59448122_3/model_epoch13.pt', num_outputs=7)
else:
    pain_detector = PainDetector(image_size=160, checkpoint_path='checkpoints/50342566/50343918_3/model_epoch4.pt', num_outputs=40)

# define a video capture object
vid = cv2.VideoCapture(0)
ret, ref_frame1 = vid.read()
print('Device: ', pain_detector.device)
# In this example the reference frames are identical, but in a real scenario, the idea is to use different
# reference frames from the same person. Ideally, the reference frames should have a neutral expression and should
# exhibit slight lighting and camera angle variations.
pain_detector.add_references([ref_frame1, ref_frame1, ref_frame1])
# Ask the user to pick a color
# r,g,b=mcolor()
r, g, b = 255, 0, 0
while (True):

    # Capture the video frame by frame
    ret, target_frame = vid.read()
    pain_estimate = pain_detector.predict_pain(target_frame)
    print("pain_estimate", pain_estimate)
    if pain_estimate > 1:
        pain_text = 'Pain'
    else:
        pain_text = 'No Pain'
    cv2.putText(target_frame, "Pain estimate : {}".format(pain_estimate), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(target_frame, "Have: {}".format(pain_text), (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('Capture frame', target_frame)

    ####################################################################################
    # Display the resulting frame
    # cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the window
