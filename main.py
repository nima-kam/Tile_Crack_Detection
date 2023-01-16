import cv2 
import numpy as np
from matplotlib import pyplot as plt
import json
import os
from utils import *

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
    return None

def rotation_matching(image,pattern):
    return None

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
    crop(img)
        
    # predict()

