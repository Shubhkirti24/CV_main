import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import math
import cv2
import sys

def NonMaxiSupr(B_area):
    
    if len(B_area) == 0:
        return []
    
    x1 = B_area[:, 0]  
    y1 = B_area[:, 1]  
    x2 = B_area[:, 2]  
    y2 = B_area[:, 3]  
    
   
    min_tres = 0.64
    AreaoftheBox = ((x2 - x1) + 1)*((y2 - y1) + 1) 
    sizeX = len(x1)
    index = np.arange(sizeX)
    for i,b in enumerate(B_area):

        current_index = index[index!=i]
      
        temp_x1 = np.maximum( B_area[current_index,0],b[0])
        temp_y1 = np.maximum( B_area[current_index,1],b[1])
        temp_x2 = np.minimum( B_area[current_index,2],b[2])
        temp_y2 = np.minimum( B_area[current_index,3],b[3])
 
        width = np.maximum(0, temp_x2 - temp_x1 + 1)
        height = np.maximum(0, temp_y2 - temp_y1 + 1)

        comOverlapArea = ((width * height) / AreaoftheBox[current_index])
        
        # Threshold implementation check , above code
        if np.any(comOverlapArea) > min_tres:
            index = index[index != i]
   
    return (B_area[index]).astype(int)

# The function to get the staff lines

def get_lines(img):
  img = np.array(img)
  img2 = np.zeros_like(img)

  # plt.imshow(img2)
  # plt.show()
  
  thresh = 230
  line_list = []
  for i in range(img.shape[0]):
    count=0
    for j in range(img.shape[1]):
        if img[i][j]>thresh:
            count += 1
    if count>500:
        for k in range(img.shape[1]):
            img2[i][k] = 255
    else:
        line_list.append(i)
  return line_list

# all_lines = get_lines(img)

def clean_lines(line_list) :
  new_list = []
  for i in range(len(line_list)-1):
    lower = line_list[i] - 3
    higher = line_list[i] + 3
    if  lower < line_list[i+1] < higher :
      i += 1
    else :
      new_list.append(line_list[i])
  new_list.append(line_list[-1])
  return new_list

# staff = clean_lines(all_lines)

def clean_centers(note_centers) :

  note_centers = sorted(note_centers, key=lambda x: x[0])
  clean_centers_list = []

  for i in range(len(note_centers)-1):
    x,y = note_centers[i]

    lower_x = x-2
    higher_x = x+2
    
    lower_y = y-5
    higher_y = y+5

    next_x, next_y = note_centers[i+1]

    if  (lower_x <= next_x <= higher_x) and (lower_y < next_y < higher_y):
      i += 1

    else :
      clean_centers_list.append(note_centers[i])
      
  clean_centers_list.append(note_centers[-1])

  return clean_centers_list



IP = Image.open(sys.argv[1]).convert('L')
temp1 = Image.open("template1.png").convert('L')
temp2 = Image.open("template2.png").convert('L')
temp3 = Image.open("template3.png").convert('L')

# Convert the images to numpy arrays
IP_arr = np.array(IP)
OP_arr = np.array(IP.convert('RGB'))
OP_arr2 = np.array(IP.convert('RGB'))
temp1_arr = np.array(temp1)
temp2_arr = np.array(temp2)
temp3_arr = np.array(temp3)
R = (255,0,0)
G =(0,255,0)
B=(0,0,255)
all_lines = get_lines(IP)
staff_lines = clean_lines(all_lines)


# Define a min_thresh for matching
min_thresh = 0.7


pattern_region_temp1 = np.zeros(IP_arr.shape)


# FInding the template 1 topleft corners in the IP image
for i in range(temp1_arr.shape[0], IP_arr.shape[0]):
    for j in range(temp1_arr.shape[1], IP_arr.shape[1]):
        sliding_window = IP_arr[i-temp1_arr.shape[0]:i, j-temp1_arr.shape[1]:j]
        cor_coeff = np.corrcoef(sliding_window.flatten(), temp1_arr.flatten())[0,1]
        if cor_coeff > min_thresh:

            pattern_region_temp1[i,j] = cor_coeff

# Matches - [N,2] => topleftcorner of Y and X coordinate
match_locs = np.argwhere(pattern_region_temp1 != 0)


# Draw a box around each match in the output image
note_centers = []
boxes = []
i=0
for loc in match_locs:
    top_left = ((loc[1]-temp1_arr.shape[1])+1, (loc[0]-temp1_arr.shape[0])+1)
    bottom_right = (loc[1], loc[0])
    x_center = int(((top_left[0] + bottom_right[0])/2))
    y_center = int(((top_left[1] + bottom_right[1])/2)-2)
    note_centers.append([x_center,y_center])
    boxes.append(np.array([top_left[0], top_left[1], bottom_right[0], bottom_right[1]]))
    i=i+1
    
    # Mark the center pixel with a ORANGE dot.
    cv2.circle(OP_arr, (x_center, y_center), 1, (255, 102, 0), -1)


boxes = np.array(boxes, dtype=int)       
boxes = NonMaxiSupr(boxes)
note_centers=[]
for it in boxes:
    x_center = int(((it[0] + it[0])/2))
    y_center = int(((it[1] + it[1])/2)-2)
    note_centers.append([x_center,y_center])

# Cleaning all the note_centers :
note_use = sorted(clean_centers(note_centers), key=lambda y: y[1])

# Convert note_use to NumPy array
note_use = np.array(note_use)
    
# Get y-coordinates
y_coordinates = note_use[:,1].tolist()


# Creating a list of staff-bins
staff_bins = list(range(5,len(staff_lines)+5 ,5))
bins = []
notes_dict = {}
# To store the value of the bass_treble clefs)
bt_list = {}
# Region indicates -> Bass or treble
region = 1 
    
for k in staff_bins:
  staff_region = staff_lines[(k-5):k]

  bin_height = int(math.ceil(np.mean(np.diff(staff_region)))/2)
  # 2 bin heights above staff
  lower_height = bin_height*2
  # 6 bin heights below staff
  upper_height = bin_height*6

  bin_range = (list(range(staff_region[0]-lower_height, staff_region[-1]+ upper_height ,bin_height)))
      
  if int(region%2) == 1 :
    sound = 'treble'
    region += 1
    dict_bin = {}
    # treble
    class_1_upper = ['E','D','F','E','D','C','B','A','G','F','E','D','C','B','A','G']
    
    for i in range(len(class_1_upper)):
      dict_bin[bin_range[i]] = class_1_upper[i]
      notes_dict[bin_range[i]] = class_1_upper[i]
      bins.append(bin_range[i])
      bt_list[bin_range[i]] = sound
  else :
    if int(region%2) == 0 :
      sound = 'treble'
      region += 1
      dict_bin = {}
      # bass
      class_2_lower = ['C','B','A','G','F','E','D','C','B','A','G','F','E','D','C','B']

      for i in range(len(class_2_lower)):
        dict_bin[bin_range[i]] = class_2_lower[i]
        notes_dict[bin_range[i]] = class_2_lower[i]
        bins.append(bin_range[i])
        bt_list[bin_range[i]] = sound

# The 'Notes_dict' contains the list of all the notes assignment

# Converting the Y Co-ordinates to bins:
result = []
for y in y_coordinates:
  if y < bins[0]:
    result.append(bins[0])
  elif y > bins[-1]:
    result.append(bins[-1])
  else:
    index = np.searchsorted(bins, y)
    result.append(bins[index]) 

# The final list to store all the notes
output_list = [notes_dict[k] for k in result]
output_list

## Creating a dictionary to store all the co-ordinates,notes and cliffs
coord_note_sound ={}

for i in range(len(result)):
  compare = result[i]
  cliff = bt_list[compare]
  note = output_list[i]
  x,y = note_use[i]
  cord = (x,y)

  coord_note_sound[cord] = (note,cliff)
t1 = boxes
for box in boxes:
    cv2.rectangle(OP_arr, box[:2], box[2:], thickness=1 ,color=R)

#FOR TEMPLATE 2

min_thresh = 0.9
pattern_region_temp2 = np.zeros(IP_arr.shape)

# FInding the template 1 topleft corners in the IP image
for i in range(temp2_arr.shape[0], IP_arr.shape[0]):
    for j in range(temp2_arr.shape[1], IP_arr.shape[1]):
        sliding_window = IP_arr[i-temp2_arr.shape[0]:i, j-temp2_arr.shape[1]:j]
        cor_coeff = np.corrcoef(sliding_window.flatten(), temp2_arr.flatten())[0,1]
        if cor_coeff > min_thresh:

            pattern_region_temp2[i,j] = cor_coeff

# Matches - [N,2] => topleftcorner of Y and X coordinate ()
match_locs_temp2 = np.argwhere(pattern_region_temp2 != 0)


# Draw a box around each matchh --> in the output image
boxes = []
i=0
for loc in match_locs_temp2:
    top_left = ((loc[1]-temp2_arr.shape[1])+1, (loc[0]-temp2_arr.shape[0])+1)
    bottom_right = (loc[1], loc[0])
    boxes.append(np.array([top_left[0], top_left[1], bottom_right[0], bottom_right[1]]))
    i=i+1
    #cv2.rectangle(OP_arr, top_left, bottom_right, thickness=2, color=B)
boxes = np.array(boxes, dtype=int)       
boxes = NonMaxiSupr(boxes)
t2 = boxes
for box in boxes:
    cv2.rectangle(OP_arr, box[:2], box[2:], thickness=1 ,color=B)




#FOR TEMPLATE 3

min_thresh = 0.6
pattern_region_temp3 = np.zeros(IP_arr.shape)

# FInding the template 1 topleft corners in the IP image
for i in range(temp3_arr.shape[0], IP_arr.shape[0]):
    for j in range(temp3_arr.shape[1], IP_arr.shape[1]):
        sliding_window = IP_arr[i-temp3_arr.shape[0]:i, j-temp3_arr.shape[1]:j]
        cor_coeff = np.corrcoef(sliding_window.flatten(), temp3_arr.flatten())[0,1]
        if cor_coeff > min_thresh:

            pattern_region_temp3[i,j] = cor_coeff

# Matches - [N,2] => topleftcorner of Y and X coordinate
match_locs_temp3 = np.argwhere(pattern_region_temp3 != 0)

# Annotations : Adding the notes to the image
for i in range(len(note_use)):
  x1, y1 = note_use[i]
  label = output_list[i]
  font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.putText(OP_arr, label, (x1,y1), font, fontScale=0.6, color = (255, 0, 0), thickness=2)


boxes = []
i=0
for loc in match_locs_temp3:
    top_left = ((loc[1]-temp3_arr.shape[1])+1, (loc[0]-temp3_arr.shape[0])+1)
    bottom_right = (loc[1], loc[0])
    boxes.append(np.array([top_left[0], top_left[1], bottom_right[0], bottom_right[1]]))
    i=i+1
    #cv2.rectangle(OP_arr, top_left, bottom_right,thickness=2 ,color=G)
boxes = np.array(boxes, dtype=int)       
boxes = NonMaxiSupr(boxes)
t3 = boxes
for box in boxes:
    cv2.rectangle(OP_arr, box[:2], box[2:], thickness=1 ,color=G)
labels = []
pitch = []

# iterate through the dictionary and append the values at index 0 and index 1 to the appropriate list
for key, value in coord_note_sound.items():
    labels.append(value[0])
    pitch.append(value[1])
# Saving it in Txt file


#coord 0,1

with open("output.txt", "w") as file:
    
    for loc,l,p in zip(t1,labels,pitch):
          top_left = ((loc[1]-temp3_arr.shape[1])+1, (loc[0]-temp3_arr.shape[0])+1)
          bottom_right = (loc[1], loc[0])
          file.write(str(top_left[0])+" "+str(top_left[1]) +" "+ str(bottom_right[0]) +" "+str(bottom_right[1]) +" "+str(bottom_right[0] - top_left[0] + 1) +" "+str(bottom_right[1] - top_left[1] + 1)+" "+l  +" MusicNote"+ "\n")
          
with open("output.txt", "a") as file:
    for loc in t2:
        top_left = ((loc[1]-temp3_arr.shape[1])+1, (loc[0]-temp3_arr.shape[0])+1)
        bottom_right = (loc[1], loc[0])
        file.write(str(top_left[0])+" "+str(top_left[1]) +" "+ str(bottom_right[0]) +" "+str(bottom_right[1]) +" "+str(bottom_right[0] - top_left[0] + 1) +" "+str(bottom_right[1] - top_left[1] + 1)+" "+ "template2"+ "\n")
with open("output.txt", "a") as file:
    for loc in t3:
        top_left = ((loc[1]-temp3_arr.shape[1])+1, (loc[0]-temp3_arr.shape[0])+1)
        bottom_right = (loc[1], loc[0])
        file.write(str(top_left[0])+" "+str(top_left[1]) +" "+ str(bottom_right[0]) +" "+str(bottom_right[1]) +" "+str(bottom_right[0] - top_left[0] + 1) +" "+str(bottom_right[1] - top_left[1] + 1)+" "+ "template3"+ "\n")

#Display the output image with the matches found
output_img = Image.fromarray(OP_arr)
save_file = "OP_"+"file.png"
output_img.save(save_file)