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
import os


main_args = live_main.parse_args([])
main_args.phase='run'
main_args.vid_dir='data/vids/baby' 
main_args.out_dir='data/output/baby'
main_args.amplification_factor=15
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

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

x1 = None

tfconfig, config = live_main.init(main_args)
with tf.Session(config=tfconfig) as sess:
    
    # grab a pointer to the video stream and initialize the FPS counter
    print("[INFO] sampling frames from webcam...")
    #vs = WebcamVideoStream(src=0).start()
    print(args)
    is_live = args.video == ""
    video_name = os.path.basename(args.video) if args.video != "" else ""
    if args.video != "" :
        stream = cv2.VideoCapture(args.video)
    else:
        stream = cv2.VideoCapture(0)

    model = live_main.init_session(main_args, sess, config, True, [100, 203])
    #fps = FPS().start()
    lastframe = None
    frame_number = 6
    mark_time = 10 * 25 # 10s * 25 frame/s
    frame_interval = mark_time / frame_number
    index = 1
    interval_count = 1
    # loop over some frames
    while stream.isOpened() : #True: #fps._numFrames < args["num_frames"]:
        start = timer()

        (grabbed, frame) = stream.read()
        if not is_live:
            stream.set(1, frame_number * index - 1)
            index = index + 1

        if not grabbed :
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Draw the rectangle around each face
        #for (x, y, w, h) in faces:
        #    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        if len(faces) > 0 :
            (x, y, w, h) = faces[0]
            #cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            x0 = x - w
            x0 = 0 if x0 < 0 else x0
            y0 = y + h
            y0 = frame.shape[0]-1 if y0 >= frame.shape[0] else y0
            w0 = 3 * w if x + 2 * w < frame.shape[1] else frame.shape[1] - x0
            h0 = h
            if x1 is None:
                (x1,y1,w1,h1)=(x0,y0,w0,h0)
        print(x1,y1,w1,h1)
        if True:
            if x1 is not None:
                # if w1 > 203 :
                #     frame2 = imutils.resize(frame[y1:y1+h1,x1:x1+w1,:], height=100, width=203)  # 400, 203, 100
                # else:
                #     frame2 = frame[y1:y1+h1,x1:x1:w1,:]
                #frame2 = imutils.resize(frame[y1:y1+h1,x1:x1+w1,:], height=90, width=203)  # 400, 203, 100
                frame2 = frame[y1:y1+h1,x1:x1+w1,:]
                print(frame2.shape)
                #frame2 = imutils.resize(frame2, height=24, width=100)
                frame2 = cv2.resize(frame2, (160,40), interpolation=cv2.INTER_AREA) # 100,24 160,40
                print(frame2.shape)
            else:
                frame2 = imutils.resize(frame, height=90, width= 203)
        else:        
            frame2 = imutils.resize(frame, width=100)
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
            mark_interval = False
            if not is_live:
                if index > frame_interval * interval_count :
                    mark_interval = True
                    interval_count = interval_count + 1
                else:
                    mark_interval = False

            w2 = int(newframe.shape[1]/2 * 0.5)
            h2 = int(newframe.shape[0]/2 * 1.5)
            pulseidx1,pulseimg1 = pulse_image.append_pulse(pulseimg1, newframe, 
                [0,frame2.shape[1]], 
                [h2, h2+1], pulseidx1, mark_interval, True)
            pulseidx1 = pulseidx1+1
            pulseidx2,pulseimg2 = pulse_image.append_pulse(pulseimg2, newframe, 
                [w2, w2+1], 
                [0,frame2.shape[0]], pulseidx2, mark_interval, True)
            pulseidx2 = pulseidx2+1

            old1=newframe.shape[1]
            newframe = imutils.resize(newframe, width=frame.shape[1])
            scale = frame.shape[1] / old1

            if True :#args["display"] > 0:
                cv2.rectangle(frame, (x1, y1), (x1+w1,y1+h1), (0,255,0))
                cv2.imshow(video_name + " Frame", frame)
                
                cv2.line(newframe,(0, int(h2 * scale)),(newframe.shape[1],int(h2*scale)), (0,255,0))
                cv2.line(newframe,(int(w2 * scale), 0),(int(w2*scale),newframe.shape[0]), (255,0,0))
                cv2.imshow(video_name + " MotionMag Frame", newframe)

                # pulseimg1show = imutils.resize(pulseimg1, height=50)
                # pulseimg2show = imutils.resize(pulseimg2, height=50)
                # cv2.imshow("Pulse 1 Hor Line", pulseimg1show)
                # cv2.imshow("Pulse 2 Ver Line", pulseimg2show)
                cv2.imshow(video_name + " Pulse 1 Hor Line", pulseimg1)
                cv2.imshow(video_name + " Pulse 2 Ver Line", pulseimg2)
                
        # update the FPS counter
        lastframe = frame2
        #fps.update()
        end = timer()
        print(end - start)
        
        key = cv2.waitKey(1)
        if key != -1:
            break 
    if pulseimg1 is not None:
        cv2.imwrite(video_name + "pulse1.png", pulseimg1)
    if pulseimg2 is not None:
        cv2.imwrite(video_name + "pulse2.png", pulseimg2)
    key = cv2.waitKey(0)
    
    stream.release()
    cv2.destroyAllWindows()
    #vs.stop()

