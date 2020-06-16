
A Real time social distance identification tool from a video or image.

Even though millions of people are staying at home to help flatten the curve,
many people in different industries are still having to go to work everyday.

So in order to monitor such case a real time social distancing detection can be implemented.
To complement and to help ensure social distancing protocol i had developed an AI-enabled social distancing detection tool that can detect 
if people are keeping a safe distance from each other by analyzing real time video streams from the camera.

END to END steps are mentioned below:

Steps of approach:

1)Object Detection using YoloV3 pretrained model.

2)finding each centroids of the detected boxes.

3)calculate the distance between different pairs of centroids.

4)if distance is less than thresold value then it is a violation,else safe.

Note: The distance taken is the pixel distance and is not calibrated according to the view.

Future developments include the following:

1)Calibaration and distance adjustment according to the viewing angle.

2)Provide a birdseye view so that density of the whole area can be known.
