import cv2 
import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure
from skimage.exposure import match_histograms
from skimage.feature import local_binary_pattern
import json
import os
from utils import *
from scipy import ndimage

# enable pyplot interactive mode for showing images
# plt.ion()

constants ={
    "image_size":(1200,1200),
    "image_folder":"images/",
    "label_name":"test_img",
    "kernel_size":9,
    'pattern_name':"AYLIN.tif",
    
}

def crop(image, width=None, height=None):

    grayscale = to_grayscale(image)
    # imshow(grayscale)  
    blurred = blur(grayscale,9)
    # imshow(blurred)
    # edges = to_edges(blurred,40,100)      
    # imshow(edges)  

    bi = to_binary(blurred,otsu=False,thresh=120)# threshold should be dynamic
    # imshow(bi)  

    kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,(7,7)
    )
    op = opening(bi,kernel)    
    # imshow(op)  

    vertices = find_vertices(op).squeeze()


    # visualize the result of corner finding
    # imshow(image,False)    
    # plt.scatter([x for x, y in vertices], [y for x, y in vertices])
    # plt.show()
    
    if len(vertices) ==4:
        if width is None or height is None:
            size = None 
        else:
            size = (width, height)
        crop_image=crop_out(image,vertices=vertices,size=size)
    imshow(crop_image)    

    return crop_image

def histogram_matching(image,pattern):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    pattern = cv2.cvtColor(pattern,cv2.COLOR_BGR2RGB)
    matched = match_histograms(image, pattern ,
                           multichannel=True)

    # visualize the result of histogram matching
    plt.imshow(matched)
    plt.show()

    return matched

def rotation_matching(image,pattern):
    pattern = cv2.cvtColor(pattern,cv2.COLOR_BGR2RGB)
    pattern = pattern.astype(float)

    # resize image to pattern dimensions
    minimum = np.min([image.shape[1] , image.shape[0]])
    resized_pattern = cv2.resize(pattern, (minimum , minimum), interpolation = cv2.INTER_AREA).astype(float)
    image = cv2.resize(image, (minimum , minimum), interpolation = cv2.INTER_AREA).astype(float)

    diff = image - resized_pattern
    min_error = np.sum(np.abs(diff))
    temp = image.copy()
    rotated_image = image.copy()


    # for each rotation ( theta ) possible ...
    for i in range(0,4):
        # now subtract the image from the pattern ....
        # calculate the error with this angle ...
        temp = cv2.rotate(temp,cv2.ROTATE_90_CLOCKWISE,temp)

        #diff = cv2.subtract(temp, pattern)
        diff = temp - resized_pattern
        # plt.imshow(diff/255)
        # plt.show()
        err = np.sum(np.abs(diff))

        if err < min_error :
            min_error = err
            rotated_image = temp.copy()
            # plt.imshow(temp/255)
            # plt.show()
            # print(i)
     
    
    # print mse and rotation angle
    print(f"mse between image and its pattern {min_error} ")


    #rotation angle in degree
    #image = ndimage.rotate(image, rotation_theta)

    plt.imshow(rotated_image/255)
    plt.show()

    return rotated_image

def binary_threshold(image):
    return None


def lbp(image):
    METHOD = 'ror'
    lbp_image = local_binary_pattern(image, 8, 1, METHOD)
    return lbp_image


def predict(img, pattern):
    """
    :params: img: input RGB Tile image
    :params: pattern: input RGB Tile pattern
    """

    crop_img=crop(image=img)
    matched_img=histogram_matching(image=crop_img,pattern=pattern)
    rotated_img=rotation_matching(image=matched_img,pattern=pattern)
    # bi_img=binary_threshold(rotated_img)
    
    return None

if __name__ == "__main__":
    """
    for testing the function result by runnig the program
    """
    path = os.path.join(os.path.dirname(constants["image_folder"]),constants["label_name"])
    img_path = path + ".jpg"
    label_path = path + ".json"
    print(path,f"\n{img_path},{label_path}")
    img = cv2.imread(img_path)
    # imshow(img)
    # plt.show()

    # open box label json file
    f = open(label_path, encoding="utf8")
    data = json.load(f)
    f.close()

    pattern = cv2.imread(constants["pattern_name"])        
    print(f"json label: {data}\n\nimage shape: {img.shape}\n\npattern shape: {pattern.shape}")
    
    # test crop
    img = crop(img)

    
    # test historgam matching 
    matched = histogram_matching(img,pattern)   

    # rotated for matched image...
    # his_matched_rotated = rotation_matching(matched,pattern) 

    # rotated image...
    rotated = rotation_matching(img,pattern) 
    # predict()

    #rotated = cv2.resize(rotated, (rotated.shape[1] , rotated.shape[0]), interpolation = cv2.INTER_AREA).astype(float)

    gs_rotated = to_grayscale(rotated.astype(np.uint8)) 
    gs_pattern = to_grayscale(pattern.astype(np.uint8))   

    rotated_lbp = lbp(gs_rotated)
    imshow((rotated_lbp),False)
    

    imshow(lbp(gs_pattern))
    plt.show()

    # print(rotated_lbp.shape)
    # print(gs_pattern.shape)

    gs_pattern_resized = cv2.resize(gs_pattern, rotated_lbp.shape, interpolation = cv2.INTER_AREA).astype(float)

    imshow((rotated_lbp),False)
    imshow(lbp(gs_pattern_resized))

    rotated_lbp_uint = np.uint8(rotated_lbp)
    # rotated_morph = cv2.morphologyEx(rotated_lbp_uint,cv2.MORPH_OPEN,(3,3),iterations=2)
    # imshow(rotated_morph)
    # plt.show()
    blur_rotated_morph = blur_freq(rotated_lbp_uint,65)
    imshow(blur_rotated_morph)
    plt.show()

    diff = blur_rotated_morph - gs_pattern_resized
    imshow(diff)
    plt.show()
    diff2 = gs_pattern_resized - blur_rotated_morph
    for i in range(1000):
        diff2 = diff2 - blur_rotated_morph

    imshow(diff2)
    plt.show()

    diff2 = 255 - diff2
    imshow(diff2)
    plt.show()

    diff3 = diff2
    # diff3 = cv2.morphologyEx(diff2,cv2.MORPH_CLOSE,(5,5),iterations=10)
    # imshow(diff3)
    # plt.show()
    plt.show()

    gs_pattern_resized2 = cv2.resize(gs_pattern, rotated_lbp.shape, interpolation = cv2.INTER_AREA).astype(float)
    kernel = np.ones((5, 5), np.uint8)
    gs_pattern_resized2 = cv2.erode(gs_pattern_resized2, kernel,iterations=2)
    opening = cv2.morphologyEx(gs_pattern_resized2, cv2.MORPH_OPEN,
                           kernel, iterations=2) 
    imshow(opening)
    plt.show()

    diff4 = opening - diff3
    for i in range(1000):
        diff4 = opening - diff4
    
    imshow(diff4)
    plt.show()
    # pattern2 = 255 - gs_pattern_resized
    # pattern2 = cv2.morphologyEx(pattern2,cv2.MORPH_DILATE,(5,5),iterations=171)
    # imshow(pattern2)
    # plt.show()
    # for i in range(500):
    #     diff2 = diff2 - pattern2
    
    # imshow(diff2)
    # plt.show()