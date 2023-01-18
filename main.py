import cv2 
import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure
from skimage.exposure import match_histograms
import json
import os
from utils import *
from scipy import ndimage

# enable pyplot interactive mode for showing images
# plt.ion()

constants ={
    "image_size":(1200,1200),
    "image_folder":"images/",
    "label_name":"1644437322.3866885",
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


    # visualize the result
    imshow(image,False)    
    plt.scatter([x for x, y in vertices], [y for x, y in vertices])
    plt.show()
    
    if len(vertices) ==4:
        if width is None or height is None:
            size = None 
        else:
            size = (width, height)
        crop_image=crop_out(image,vertices=vertices,size=size)
    imshow(crop_image)    

    return crop_image

def histogram_matching(image,pattern):
    image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    pattern = cv2.cvtColor(pattern,cv2.COLOR_BGR2RGB)
    matched = match_histograms(image, pattern ,
                           multichannel=True)
    plt.imshow(matched)
    plt.show()
    return matched

def rotation_matching(image,pattern):
    # resize image to pattern dimensions
    resized = cv2.resize(image, (pattern.shape[0] , pattern.shape[1]), interpolation = cv2.INTER_AREA)

    h, w , d = resized.shape
    rotation_theta = 0

    diff = cv2.subtract(resized, pattern)
    min_error = np.sum(np.abs(diff))

    # for each rotation ( theta ) possible ...
    for i in range(0,359,90):
        # now subtract the image from the pattern ....
        # calculate the error with this angle ...
        diff = cv2.subtract(resized, pattern)
        err = np.sum(np.abs(diff))

        if err < min_error :
            min_error = err
            rotation_theta = i
     
    
    # print mse and rotation angle
    print(f"mse between image and its pattern {min_error} and theta ( rotation angle ) : {rotation_theta}")


    #rotation angle in degree
    image = ndimage.rotate(image, rotation_theta)
    return image

def binary_threshold(image):
    return None


def predict(img, pattern):
    """
    :params: img: input RGB Tile image
    :params: pattern: input RGB Tile pattern
    """

    crop_img=crop(image=img)
    matched_img=histogram_matching(image=crop_img,pattern=pattern)
    rotated_img=rotation_matching(image=matched_img,pattern=pattern)
    bi_img=binary_threshold(rotated_img)
    
    return None

if __name__ == "__main__":
    """
    for testing the function result by runnig the program
    """
    path = os.path.join(os.path.dirname(constants["image_folder"]),constants["label_name"])
    img_path = path + ".png"
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

    # rotated ...
    rotated = rotation_matching(matched,pattern) 
    # predict()

