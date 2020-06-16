

'''setting yp the configuration'''

model_path = "model_yolo"

#applying minimum confidence and threshold filters when applying non max supression(NMS)
min_conf = 0.4
nms_thresh = 0.4

#To check for the use of gpu or not 
use_gpu = False

#set minimum safe distance(pixel distance)
min_dist = 50