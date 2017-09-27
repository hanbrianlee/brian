# **Finding Lane Lines on the Road** 

## BRIAN LEE (HAN UL LEE)


---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on my work in a written report

The code can be found in CarND-LaneLines-P1/P1.ipynb in my github repository.

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps.
1. Grayscale conversion
2. Gaussian smoothing/blurring
3. Canny Edge Detection
4. Masking off regions outside region of interest
5. Hough transform and drawing lines over lanes

In order to draw a single line on the left and right lanes, I modified the draw_lines() function quite heavily. The code does need some clean-up such as shorter and clearer variable names and removal of unnecessary comment lines or lines written for debugging purposes. However, I left them as much as they were written, in order to refresh my memory later on when I look back to it and also to show the process of my work in finding the solution.

First of all, the draw_lines() function's cv2.line function needed to be modified. The code in the existing helper function for draw_lines was meant to draw lines for every lines detected by hough transform lines finding algorithm. This would produce a result similar to that given in examples/raw-lines-example.mp4. However, since this is not what we are ideally looking for, the cv2.line function needed to be changed so that it draws only one line for the left lane and one line for the right lane. In my code, this code looks like the following:
    **Left Lane:**  cv2.line(img,(int(finalLLTx),yLaneTop),(int(finalLLBx),yLaneBot), color, thickness) 
    **Right Lane:**  cv2.line(img,(int(finalRLTx),yLaneTop),(int(finalRLBx),yLaneBot), color, thickness)
The finalLLTx indicates the x coordinate of the top point of the left line extrapolated all the way up to the top of the region of interest (with its corresponding y coordinate yLaneTop), and the finalLLBx indicates the x coordinate of the bottom point of the left line extrapolated all the way down to the bottom of the region of interest (with its corresponding y coordinate yLaneBot). Similar explanation goes for the right lane. One side thing to note is that the yLaneBot value had to be set to a value larger (so lower on the image x,y coordinate system) to go beyond the image frame, in order to have the line to truly begin from the bottom of the frame. If yLaneBot were to be set exactly to the maximum y value of the image, then the line's circular endpoint will show, which is undesirable as it does not meet the rubrics of this project. 

Also, I found out that the lines detected by the hough_lines function were not really ideal, causing the averaged slopes of the left and right lines detected respectively to deviate from each other a bit in going from one frame to the next. In order to smooth out this deviation between frames, a moving average of 3 frames' final slopes were utilized. However, doing this only still caused the drawn lines to abruptly change quite a bit. So in order to mitigate this issue, a tanh function was utilized to limit the amount of finalLLTx, finalLLBx, finalRLTx, and finalRLBx points vary from frame to frame. For example: 
finalRLTx = prevRLTx + 1*np.tanh(newRightLaneTop_x - prevRLTx)

The moving averaged slope was also applied this tanh function utilizing smoothing in order to optimally reduce abrupt changes in lines drawn. The constant number 1 in front of the np.tanh function indicates that the finalRLTx is only allowed to vary a maximum of plus 1 or minus 1 pixel from a frame to the next. Global variables were utilized to store the previous frame's information.

Finally, the hough_lines function input parameters were modified to add another image. This additional input image was assigned the original image from the main pipeline function so that the unaltered original image can be passed onto the draw_lines function. This allowed for the add_weighted function later on to more cleanly overlay the lines onto the original image without suffering intensity losses of the original image.



### 2. Identify potential shortcomings with your current pipeline

There are a couple of shortcomings my current pipeline has, namely:

1. It will fail to detect curved lanes properly as the current implementation of hough transforms only find straight lines and if the curved lanes are curvy enough, the current setting for minimum vote threshold and minimum pixels required to qualify as lines will not pick up the curved lanes as lanes.

2. The current set up for region of interest is somewhat specifically optimized for test videos and test images included as part of this project. If the camera angle were to be too high or low (in the case of challenge video where the camera is either installed quite lower compared to the test images and test videos or just pointing downwards more), then the camera image might pick up on too little of the lanes or pick up on the front hood of the car, both of which will present issues when determining lanes with hough transforms. Also, in case the car is going on a road that steeply declines then inclines (like a valley or mountaineous regions), then my region of interest settings will fail to pick up on ideal region of interest. A lane change scenario also falls under similar category of potential issues.



### 3. Suggest possible improvements to your pipeline

Possible improvements to the above shortcomings as well as potential improvements are as follows:

1. The hough transform's parameters could be changed so that it allows smaller straight line segments that would appear on curved lanes to be detected. This means that the minimum vote threshold and minimum pixels reuiqred for a line will have to be both tuned down. Of course, the region of interest will have to be modified a bit as well so that it does not pick up on images that suddenly appear ahead because of road curving to the right or left. Tuning down the above mentioned hough transform parameters, however, will cause the pipeline to pick up on a lot more of noise lines that are not part of the lanes. But considering that a slope line very close to 0 is not possible, those lines can be discarded. With those discarded, perhaps another voting system can be employed where if for the left lanes (negative slopes), if the average of the lower half and the average of the higher half are apart by a certain degree, that lane can be deemed curving. Also, depending on the polarity of the difference of the average of the lower half and the average of the higher half, one can figure out which way the lane is curving. Based on this information, the region of interest can be modified accordingly every frame, in order to better mask off the regions that are not needed for lane finding.

2. An algorithm in which region of interest settings can be automatically tuned depending on the images could be developed. The algorithm may be a supervised learning algorithm (logistical classification - utilizing neural networks if needed) that learns what roads look like from a lot of sample images, and predicts which blocks of pixels in the images correspond to roads (using varying sliding window technique). Perhaps this step could be carried out every certain number of frames of the video in order to update the region of interest, or it could be run for the whole video. However, if live video image is being fed, just like in a self-driving car, then the region-of-interesting finding algorithm would most likely have to be run every few milliseconds. Upon detecting the blocks of pixels as roads, one can use max and min functions for the blocks of pixels to locate the four corners with which to draw the vertices for region-of-interest masking.

