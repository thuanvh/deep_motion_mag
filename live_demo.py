# import the necessary packages
from __future__ import print_function
from imutils.video import WebcamVideoStream
from imutils.video import FPS

import argparse
import imutils
import cv2

import live_main
import tensorflow as tf

from timeit import default_timer as timer
import numpy as np

import pulse_image


main_args = live_main.parse_args([])
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
ap.add_argument("-video", dest="video", type=str, default="", help="Video file, empty if use camera")

ap.add_argument("-output", dest="output", type=str, default="", help="Output folder")
args = ap.parse_args()

pulseimg1 = None
pulseimg2 = None
pulseidx1 = 0
pulseidx2 = 0
pulsewidth = 500

tfconfig, config = live_main.init(main_args)
with tf.Session(config=tfconfig) as sess:
    model = live_main.init_session(main_args, sess, config)
    # grab a pointer to the video stream and initialize the FPS counter
    print("[INFO] sampling frames from webcam...")
    #vs = WebcamVideoStream(src=0).start()
    print(args)
    if args.video != "" :
        stream = cv2.VideoCapture(args.video)
    else:
        stream = cv2.VideoCapture(0)
    #fps = FPS().start()
    lastframe = None
    frame_number = 9
    index = 1
    # loop over some frames
    while stream.isOpened() : #True: #fps._numFrames < args["num_frames"]:
        start = timer()

        (grabbed, frame) = stream.read()
        stream.set(1, frame_number * index - 1)
        index = index + 1

        if not grabbed :
            break 
        frame2 = imutils.resize(frame, width=203)  # 400, 203, 100
        #frame2 = imutils.resize(frame, width=100)
        #frame2 = frame

        if pulseimg1 is None:
            pulseimg1 = np.zeros((frame2.shape[1], pulsewidth, 3), np.uint8)
        if pulseimg2 is None:
            pulseimg2 = np.zeros((frame2.shape[0], pulsewidth, 3), np.uint8)

        #print(frame2.shape)
        # check to see if the frame should be displayed to our screen
        if lastframe is not None:
            start1 = timer()
            newframe = model.run_live_process(lastframe, frame2, main_args.amplification_factor)
            end1 = timer()
            print("inference:",end1-start1)

            #pulse

            w2 = int(newframe.shape[1]/2 * 0.5)
            h2 = int(newframe.shape[0]/2 * 1.5)
            pulseidx1 = pulse_image.append_pulse(pulseimg1, newframe, 
                [0,frame2.shape[1]], 
                [h2, h2+1], pulseidx1)
            pulseidx1 = pulseidx1+1
            pulseidx2 = pulse_image.append_pulse(pulseimg2, newframe, 
                [w2, w2+1], 
                [0,frame2.shape[0]], pulseidx2)
            pulseidx2 = pulseidx2+1

            newframe = imutils.resize(newframe, width=frame.shape[1])
            if True :#args["display"] > 0:
                cv2.imshow("Frame", frame)
                cv2.imshow("MotionMag Frame", newframe)
                cv2.imshow("Pulse 1 Hor Line", pulseimg1)
                cv2.imshow("Pulse 2 Ver Line", pulseimg2)
                
        # update the FPS counter
        lastframe = frame2
        #fps.update()
        end = timer()
        print(end - start)
        
        key = cv2.waitKey(1)
        if key != -1:
            break 
    if pulseimg1 is not None:
        cv2.imwrite("pulse1.png", pulseimg1)
    if pulseimg2 is not None:
        cv2.imwrite("pulse2.png", pulseimg2)
    stream.release()
    cv2.destroyAllWindows()
    #vs.stop()

