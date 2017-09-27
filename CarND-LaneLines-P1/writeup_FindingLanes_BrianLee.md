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

In order to draw a single line on the left and right lanes, I modified the draw_lines() function quite heavily.




![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline

There are a couple of shortcomings my current pipeline has, namely:
1. It will fail to detect curved lanes properly as the current implementation of hough transforms only find straight lines and if the curved lanes are curvy enough, the current setting for minimum vote threshold and minimum pixels required to qualify as lines will not pick up the curved lanes as lanes. 2. The current set up for region of interest is somewhat specifically optimized for test videos and test images included as part of this project. If the camera angle were to be too high or low (in the case of challenge video where the camera is either installed quite lower compared to the test images and test videos or just pointing downwards more), then the camera image might pick up on too little of the lanes or pick up on the front hood of the car, both of which will present issues when determining lanes with hough transforms. Also, in case the car is going on a road that steeply declines then inclines (like a valley or mountaineous regions), then my region of interest settings will fail to pick up on ideal region of interest. A lane change scenario also falls under similar category of potential issues.



### 3. Suggest possible improvements to your pipeline

Possible improvements to the above shortcomings as well as potential improvements are as follows:
1. 
2. An algorithm in which region of interest settings can be automatically tuned depending on the images could be developed. The algorithm may be of a 

Machine learning - supervised learning - find roads this way. 

Another potential improvement could be to ...
