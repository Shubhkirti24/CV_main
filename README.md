
# Part 1

### Aim
The goal for this step was to take a specific input image and create a method to denoise it. The provided image to denoise was a "mysteriously noisy" picture of a handsome bird. The additional challenge was to identify the particular noise casuing factor and after finding a way to localize the region.  Initially we tried a mean filter approach to denoise but that didnt provide good enough results so we shifted to using fourier transform to smooth out the image.

### Given data
A Noisy image obtained due to image transformations.

![alt_text](https://github.com/Shubhkirti24/CV_main/blob/main/Part_1/noisy_pichu.png)

### Implementation
For part 0 we our implementation starts by taking the input image and then grey scale it. Then take the fourier transform of that image and visualize it in the frequency domain. In there you would be able to see the noise which causes the distortions in the supposed image. Apply a patch to the area to cover the noise in the original image after manually trying to approximate its position. 


Cause of Noise - The frequency content of an image can be altered through transformations including scaling/rotation.
The Fourier domain may become noisier as a result of this shift in frequency content. The spatial frequency content of an image is changed during transformation, and additionally new spatial frequencies may also be added to that. 

# Remedy explained:
cropped out the obtained noisy area in the image to get the original image. - can be explained as manual cropping

![alt_text](https://github.iu.edu/cs-b657-sp2023/shubpras-rifhall-arsivak-kaur94-a1/blob/4cc3acf9406024ae10f8c1e6bfe5cc099c1875ae/part0/fourier_noise.png)

Tricky Part:
Prior to using the maunal [crop-out] method, mean or gaussian filtering was tried in order to determine the best acceptable method to eliminate the current noise. The goal was to identify the source and region of the noise in the fourier domain. 

## Other approaches we tried:
The image quality was reduced by the median/Gaussian filtering used to reduce noise. 

### Result:
![alt_text](https://github.com/Shubhkirti24/CV_main/blob/main/Part_1/fourier_noise.png)





# Part 2

### Aim
To detect staves (combination of 5 parallel lines) in an image and highlight it in red in the original image.

### Given data
An input image that consists of staves and some noise in form of text, etc.

### Implementation

Initial approach:
Tried applying hough transformation on the input image but it was taking too long to execute.

Second approach:
After some research, we came across an algorithm called canny edge detection which can be used to get rid of unnecessary features in an image which significantly helps in reducing the size of the data that needs to be processed. It has following steps:
1. Noise reduction: In code we have used gaussian filter to achieve this step. Gaussian filter smoothens the image and hence helps in getting rid of some of the noise.
2. Calculation of gradient: In this step we calculated x and y derivative of the image(Derivative of an image is simply the difference between adjacent pixels which gives us the change in intensity). Using the x and y derivative of the image we calculated the gradient direction which gives us the direction of the edges which are required in the next step.
3. Non-maximal suppression: Here we compare the value of every pixel with its neighboring 2 pixels in either of these directions: horizontal, vertical or diagonal. The direction in which we look for neighbors is determined by the gradient direction matrix. After determining the neighbors, we compare the values with the current pixel, if the current pixel is the largest of all, we keep that value else set it to 0. This helps in thinning out the edges and hence reducing noise.
4. Hystersis thresholding: This is the last step in canny edge detection and it is done to make edges more prominent. It divides the image matrix obtained from previous step into three kinds of pixels - irrelevant, weak and strong. Whether a pixel is irrelevant, weak or strong is decided based on some threshold value. We keep the strong pixels and only those weak pixels which has atleast one strong pixel as its neighbor.

After performing the canny edge detection, we performed hough transform. It followed the following steps:
1. First we created an array for rho and theta values each. Rho values ranges from negative of diagonal length of input image to diagonal length of input image and theta values ranges from -90 to 90. 
2. Then we created the hough matrix with rho as row index and theta as column index and initialized all the values to 0. 
3. After this we iterated through all the pixels in the image and if the value is greater than a particular threshold (in our case 200), we incremented all the cells that correspond to the respective rho and theta values (calculated by substituting in the equation: rho = x.cos(theta) + y.sin(theta)).
4. Next, we picked the top 15 peaks in the hough matrix and calculated corresponding x and y coordinates and using that plotted the lines on the original image to identify the staff lines.

### Result:
We were able to detect the staff lines in the test image given.<br />
This was the original image:<br />
<img width="261" alt="sample-input" src="https://media.github.iu.edu/user/20858/files/00b4a9d0-9d16-459b-a845-f5e50ab47059"><br />
This is the output image (after going through the algorithm, staff lines are detected by highlighting in red color):<br />

![detected_staff](https://github.com/Shubhkirti24/CV_main/blob/main/Part_2/detected_staff.png)

### Observations:
1. Canny edge detection significantly reduces the noise in the data and thus helps in easier identification of lines using hough transformation.
2. If we increase the number of peaks to be considered in hough transformation code, it detect extra lines (not staff lines) that are irrelevant to the solution.


### References:
1. https://medium.com/@ceng.mavuzer/canny-edge-detection-algorithm-with-python-17ac62c61d2e
2. https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
3. https://alyssaq.github.io/2014/understanding-hough-transform/



# Part 2

## Aim
1. To detect the notes using the given templates make and make boxes around them for easier identification
2. To classify the notes and annotate the same with Treble and Bass clefs.

### Given data
4 input images of varied natures. Images 2,3 and 4 were noisy (and even warped), hence harder to classify.

### Implementation

Initial approach:
Our inital approach included implementing a mask on the fourier images to detect the staff lines using hough transfrom. This approach led us to identify the lines, but the trnasformations took too long to implement every time.


Second approach:

#### Detection of Staff lines

1. After some research we used a code to identify the lines in the image.
We thresholded the image and created a binary image where all pixels with intensity greater than a threshold value (thresh = 230) were set to white (255), and all other pixels are set to black (0).

2. Then, we looped over each row of the binary image and counted the number of white pixels in each row. If the number of white pixels in a row was greater than 500, it was assumed to be a staff line and all pixels in that row are set to white in the binary image.

3. Finally, a list was populated with the indices of rows that were not identified as staff lines. The resulting binary image only had the staff lines in white against a black background.

#### Template matching

4. A sliding window kernel of the template size was implemented to match the first template to the image, which represents a notehead.
A nested for-loop is used to iterate over each pixel of the input image, starting from the bottom-right corner of the template image. For each pixel, a window is extracted from the input image with the same shape as the template image.
	
5. The correlation coefficient between the flattened window and the flattened template image is calculated using the np.corrcoef method, and is stored in the corresponding pixel of the corr array.
	
6. If the correlation score is greater than the threshold,it is considered a match. The locations of the matches are found and for each match found, it marks the center pixel of the notehead in the output image with an orange dot and saves the bounding box of the notehead in a list. The note_centers are store in a list.
	
7. A non-maximum suppression algorithm is applied to eliminate overlapping bounding boxes of the noteheads.
    
8. It uses the remaining bounding boxes of the noteheads to draw boxes around the noteheads in the output image.

9. Similar steps of cross-correlation coefficient for template matching and NMS algorithms are run to detect the template2.png and template3.png

# Example of template matching (intermediate result for detection) result in image2

![alt_text](https://github.com/Shubhkirti24/CV_main/blob/main/Part_3/OP_music2.png)

#### Note Classification

10. Once all the boxes are identified and the note centers are stored in a list, we implement clean_lines and clean_centers function to get rid of the overlapping/nearby points. We keep an error range of 3-5 pixels to get the actual centers of the boxes. Similary, we use clean_lines to get only a subset of the detected lines from the images.

11. Using the maximum and minimum value of the co-ordinates bins were calculated to divide the image into 'regions'. (For instance, the first 5 lines were identified as 'treble', the next '5' as 'bass' and so on alternately)

12. Within each 'region' the bins were calculated to using the average height between staff lines in that region.

13. To get the notes above and below the staff region, more bins were added (2 to the top and 6 to the bottom), with note repetitions as given by the music theory.

14. Using the sorted values of the 'Y-coordinates' of the centers, each of the co-ordinates were assigned to their resprctive bins.

15. The final 'output list' contains the assigned 'Notes' for each of the co-ordinates.

16. A final dictionary 'coord_note_sound' is created to store the list of all the co-ordinates of the identified notes with their respective clefs.


### Result:

The code identifies the notes with varied measures of success. Especially the notes between the staff lines were harder to identify and annotate.

### Observations:

- Template matching : The use of non-maximum suppression algorithm was essential in reducing the overlaps of the boxes to match the templates.

- Note Annotation : Annotating/Classifying the notes was a harder task. We would love to see how CNN / SVM perfroms on the dataset, specially for the warped images.

# Final Output obtained:
![alt_text](https://github.com/Shubhkirti24/CV_main/blob/main/Part_3/OP_file.png)
### References:

- https://pillow.readthedocs.io/en/stable/reference/
- https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/hough_lines/hough_lines.html
- https://omr-research.github.io/

