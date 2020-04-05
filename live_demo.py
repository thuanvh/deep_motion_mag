# import the necessary packages
from __future__ import print_function
from imutils.video import WebcamVideoStream
from imutils.video import FPS

import argparse
import imutils
import cv2

import live_main
import tensorflow as tf

main_args = live_main.parse_args()
main_args.phase='run'
main_args.vid_dir='data/vids/baby' 
main_args.out_dir='data/output/baby'
main_args.amplification_factor=10
# if [ "$DYNAMIC_MODE" = yes ] ; then
#     FLAGS="$FLAGS"" --velocity_mag"
# fi
main_args.config_file='configs/o3f_hmhm2_bg_qnoise_mix4_nl_n_t_ds3.conf'

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=100,
	help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=-1,
	help="Whether or not frames should be displayed")
args = vars(ap.parse_args())


tfconfig, config = live_main.init(main_args)
with tf.Session(config=tfconfig) as sess:
    model = live_main.init_session(main_args, sess, config)
    # grab a pointer to the video stream and initialize the FPS counter
    print("[INFO] sampling frames from webcam...")
    stream = cv2.VideoCapture(0)
    fps = FPS().start()
    lastframe = None

    # loop over some frames
    while True: #fps._numFrames < args["num_frames"]:
        # grab the frame from the stream and resize it to have a maximum
        # width of 400 pixels
        (grabbed, frame) = stream.read()
        frame = imutils.resize(frame, width=400)
        # check to see if the frame should be displayed to our screen
        if lastframe is not None:
            newframe = model.run_live_process(lastframe, frame, main_args.amplification_factor)
            #
            if True :#args["display"] > 0:
                cv2.imshow("Frame", frame)
                cv2.imshow("MotionMag Frame", newframe)
                key = cv2.waitKey(1) & 0xFF
        # update the FPS counter
        lastframe = frame
        #fps.update()
    # # stop the timer and display FPS information
    # fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
stream.release()
cv2.destroyAllWindows()

# def livevideo():


# if __name__ == "__main__":
#     livevideo
    