# USAGE
# python detect.py --input pedestrians.mp4


# import the necessary packages
from config import config
from detect import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os

#constructing the argument parse to parse the required arguments..

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-i", "--input", type=str, default="")
arg_parser.add_argument("-o", "--output", type=str, default="")
arg_parser.add_argument("-d", "--display", type=int, default=1)
args = vars(arg_parser.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([config.model_path,"coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
print(type(LABELS))


# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.model_path, "yolov3.weights"])
configPath = os.path.sep.join([config.model_path, "yolov3.cfg"])

# loading our YOLO object detector trained on COCO dataset with 80 classes
print("loading yolo pretrained model")

#creating a network using the yolo weights file ()
network = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

if config.use_gpu:
    print("Use CUDA for perfoming the activity")
    network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the *output* layer names that we need from YOLO
layer_names = network.getLayerNames() 
output_layers=[layer_names[i[0] - 1] for i in network.getUnconnectedOutLayers()] 

print(output_layers)

#gettting 82,94,106 as output layers

# initialize the video stream and pointer to output video file
print("Accessing the input video stream")

video_input = cv2.VideoCapture(args["input"] if args["input"] else 0)

print('Accessed')
vid_writer = None

while True:
    #looping over to read each and every frame in video
    while  True:
        (check_input, frame) = video_input.read()
        #just to check whether we are able to capture frames or not
        print('Is frame captured??:::',check_input)
        # if the frame was not captured, then we have reached the end of the video stream
        if not check_input:
            break
    
    # resizing the frame and setting the label only to person as we need only person detection
    frame = imutils.resize(frame, width=700)
    results_all = detect_people(frame, network, output_layers,
                            personIdx=LABELS.index("person"))
    #Initializing the set of violation indexes who doesnt maintain social distance.
    violation_indexes = set()

#if length of detections are greater than 2 we will compute the pair wise distances.
    if len(results_all) >= 2:
        #1)extracting the centroids
        #2)calc euclidean distances between all pairs of the centroids
        centroids = np.array([x[2] for x in results_all])
        Dist = dist.cdist(centroids, centroids, metric="euclidean")

        for i in range(0, Dist.shape[0]):
            for j in range(i + 1, Dist.shape[1]):
                #we proposed min dist of 60
                if Dist[i, j] < config.min_dist:
                    # updating our violation indexes
                    violation_indexes.add(i)
                    violation_indexes.add(j)

    # loop over the results_all
    for (i, (conf, bbox, centroid)) in enumerate(results_all):
        # extract the bounding box and centroid coordinates, then
        # initialize the color of the annotation
        (xmin, ymin, xmax, ymax) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)

        # color all violation indexes with red.
        if i in violation_indexes:
            color = (0, 0, 255)

        # draw (1) a bounding box around the person and (2) the centroid coordinates of the person,
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 1)

    # draw the total number of social distancing violations on theoutput frame
    text = "Total Violations: {}".format(len(violation_indexes))
    text_a="Total safe distant:{}".format(len(results_all)-len(violation_indexes))
    
    cv2.putText(frame, text, (10, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
    cv2.putText(frame, text_a,(10, frame.shape[0] - 80),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),3)

    # check to see if the output frame should be displayed to ourscreen
    if args["display"] > 0:
        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

    #break from the loop if key 'c' was pressed.
    if key == ord("c"):
        break

    # if an output video file path has been supplied and the video writer has not been initialized, do so now
    if args["output"] != "" and vid_writer is None:
        # (set my video codec which is four characters include MJPG , DIVX , and H264 etc...)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        #setting fps to 30 or 25 ,writing frames to video
        vid_writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)

    # if the video writer is not None, write the frame to the output video file
    if vid_writer is not None:
        vid_writer.write(frame)