import cv2 
import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure
from skimage.exposure import match_histograms
from skimage.feature import local_binary_pattern
import json
import os
from scipy import ndimage
from tensorflow import keras
import glob

from utils import *
from const import *


# enable pyplot interactive mode for showing images
# plt.ion()


input_shape=(constants['resized_dim'],constants['resized_dim'],3) # input dimensions for conv model
model=keras.applications.mobilenet.MobileNet(include_top=False,weights='imagenet',input_shape=input_shape,classes=1) # conv model
for l in model.layers:
  l.trainable =False

def find_similarity(image,pattern,feature_extractor):
    """
    runs feature_extractor model on both pattern and
    image and returns the diffrence between thier outputs
    """
    assert(image.shape==pattern.shape)
    if len(image.shape)==3:
        image = np.array([image.astype(np.float)])
    if len(pattern.shape)==3:
        pattern = np.array([pattern.astype(np.float)])

    i_feat=feature_extractor(image) # image should have 4 dims for using in model
    p_feat=feature_extractor(pattern)
    diff=keras.losses.cosine_similarity(
    i_feat,
    p_feat,
    axis=3)
    print(f"difference shape is {diff.shape}")
    return diff.numpy().squeeze() 

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
    # imshow(image,False,"initial image")    
    # plt.scatter([x for x, y in vertices], [y for x, y in vertices])
    # plt.show()
    
    if len(vertices) ==4: # the tile must have 4 corners
        if width is None or height is None:
            size = None 
        else:
            size = (width, height)
        crop_image,transform=crop_out(image,vertices=vertices,size=size)
    #imshow(crop_image,title="croped tile")    

    return crop_image,transform

def histogram_matching(image,pattern):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    pattern = cv2.cvtColor(pattern,cv2.COLOR_BGR2RGB)
    matched = match_histograms(image, pattern ,
                           multichannel=True)

    # visualize the result of histogram matching
    #plt.imshow(matched)
    # plt.title("histogram matching")
    # plt.show()

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
    final_angle = 0
    angle = 0
    for i in range(0,4):
        # now subtract the image from the pattern ....
        # calculate the error with this angle ...
        temp = cv2.rotate(temp,cv2.ROTATE_90_CLOCKWISE,temp)
        angle += 90

        #diff = cv2.subtract(temp, pattern)
        diff = temp - resized_pattern
        # plt.imshow(diff/255)
        # plt.show()
        err = np.sum(np.abs(diff))

        if err < min_error :
            min_error = err
            rotated_image = temp.copy()
            final_angle = angle
            # plt.imshow(temp/255)
            # plt.show()
            # print(i)
     
    
    # print mse and rotation angle
    print(f"mse between image and its pattern {min_error} ")


    #rotation angle in degree
    #image = ndimage.rotate(image, rotation_theta)

    #plt.imshow(rotated_image/255)
    #plt.show()

    return rotated_image,final_angle

def binary_threshold(image):
    return None


def model_input(img,pattern,label):
    img,trans=crop(image=img)

    r_pattern = cv2.resize(pattern, img.shape[:2], interpolation = cv2.INTER_AREA) # todo : matched pattern is better than pattern ?

    rotated , angle = rotation_matching(img,r_pattern) 

    # todo: pattern must be resized
    matched_pattern=histogram_matching(image=r_pattern,pattern=img) # for using as the input to detection model

    rotated_transformed_labels = transform_labels(label,transform=trans , angle=angle) # function must get label 'shapes' list from the function input
    #imshow(show_transfered_labels(rotated.astype(np.uint8),rotated_transformed_labels),title="transformed labels")

    return (rotated , matched_pattern , rotated_transformed_labels)


def find_img_proposals(rotated , r_pattern ):
    gs_rotated = to_grayscale(rotated.astype(np.uint8)) 
    gs_pattern = to_grayscale(r_pattern.astype(np.uint8))
    # bi_img=binary_threshold(rotated_img)
    
    med_blur_gs_rotated=median_blur(gs_rotated,3)
    rotated_lbp = lbp(med_blur_gs_rotated)
    #imshow((rotated_lbp),False,"image lbp")

    bi_rot_lbp=to_binary(rotated_lbp, adaptive=True,blockSize=11,C=0)
    #imshow(bi_rot_lbp,show=False,title="Image binary lbp ")

    morph_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    open_bi_lbp=closing(bi_rot_lbp,morph_kernel).astype(np.float32)
    median_blur_open_bi_lbp = median_blur(open_bi_lbp,5)
    #imshow(median_blur_open_bi_lbp,show=False,title="Image open binary lbp ")

    bin_r_pattern = to_binary(gs_pattern.astype(np.uint8),otsu=False,thresh=240).astype(np.float32)
    reversed_bin_pattern = 1 - bin_r_pattern
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    reversed_bin_pattern = cv2.dilate(reversed_bin_pattern,iterations=8,kernel=kernel,borderType=cv2.BORDER_REPLICATE)
    #imshow(reversed_bin_pattern,False,title="reversed_bin_pattern pattern")

    diff_open_bi_lbp_pattern =median_blur_open_bi_lbp - reversed_bin_pattern 
    #imshow(diff_open_bi_lbp_pattern,title="difference between bin image and pattern")

    cracks=showCountours(rotated,diff_open_bi_lbp_pattern,threshold=3000)
    proposals = get_proposals(cracks)
    #imshow(show_proposals(rotated/255,proposals),title = "proposals")
    model_proposals = get_resized_proposals(cracks)

    return model_proposals

def train(img,pattern,label):

    (rotated , r_pattern , rotated_transformed_labels) = model_input(img,pattern,label)
    model_proposals = find_img_proposals(rotated,r_pattern)
    norm_rotated = rotated.astype(np.float32)/255
    norm_pattern = r_pattern.astype(np.float32)/255
    diff_img = find_similarity(norm_rotated,norm_pattern)
    answers = []
    for propsal in model_proposals:
        max_diff = np.max(diff_img[propsal[0]:propsal[2],propsal[1]:propsal[3]])
        if max_diff > 0 :
            answers.append(propsal*32)

    counter = 0
    for answer in answers:
        if(IOU(rotated_transformed_labels,answer)):
            counter += 1
    if len(answers) > 0:
        percision = counter / len(answers)
    else:
        percision = 1

    return  percision

def accuracy():
    sum = 0
    json_files = glob.glob(constants['image_folder']+'*.json')
    for file in json_files:
        f = open(file, encoding="utf8")
        data = json.load(f)
        f.close()
        img_path = main_constants["image_folder"]+data["imagePath"]
        img = cv2.imread(img_path)
        pattern = cv2.imread("./Patterns/"+data["pattern"]) 
        sum += train(img,pattern,data)
    return sum/len(json_files)


def predict(img, pattern):
    """
    :params: img: input RGB Tile image
    :params: pattern: input RGB Tile pattern
    """

    img,trans=crop(image=img)

    r_pattern = cv2.resize(pattern, img.shape[:2], interpolation = cv2.INTER_AREA) # todo : matched pattern is better than pattern ?

    rotated , angle = rotation_matching(img,r_pattern) 

    # todo: pattern must be resized
    matched_pattern=histogram_matching(image=r_pattern,pattern=img) # for using as the input to detection model
    r_pattern = matched_pattern

    model_proposals = find_img_proposals(rotated,r_pattern)
    norm_rotated = rotated.astype(np.float32)/255
    norm_pattern = r_pattern.astype(np.float32)/255
    diff_img = find_similarity(norm_rotated,norm_pattern)
    answers = []
    for propsal in model_proposals:
        max_diff = np.max(diff_img[propsal[0]:propsal[2],propsal[1]:propsal[3]])
        if max_diff > 0 :
            answers.append(propsal*32)
    
    
    return answers

if __name__ == "__main__":
    """
    for testing the function result by runnig the program
    """
    # path = os.path.join(os.path.dirname(main_constants["image_folder"]),main_constants["label_name"])
    # label_path = path + ".json"
    

    # # open box label json file
    # f = open(label_path, encoding="utf8")
    # data = json.load(f)
    # f.close()
    # img_path = main_constants["image_folder"]+data["imagePath"]
    # img = cv2.imread(img_path)
    # print(path,f"\n{img_path},{label_path}")
    # imshow(img)
    # plt.show()
    
    # pattern = cv2.imread("./Patterns/"+data["pattern"])      
    # print(f"json label: {data}\n\nimage shape: {img.shape}\n\npattern shape: {pattern.shape}")

    print(f'accuracy : {accuracy()}')
    
    #r_pattern = cv2.resize(pattern, img.shape[:2], interpolation = cv2.INTER_AREA)
    
    # test crop
    # img = crop(img)

    
    # # test historgam matching 
    # matched = histogram_matching(img,r_pattern)

    # # rotated for matched image...
    # # his_matched_rotated = rotation_matching(matched,pattern) 

    # # rotated image...
    # rotated = rotation_matching(img,pattern) 
    # # predict()

    # #rotated = cv2.resize(rotated, , interpolation = cv2.INTER_AREA).astype(float)
    # r_pattern = cv2.resize(pattern, rotated.shape[:2], interpolation = cv2.INTER_AREA)
    # gs_rotated = to_grayscale(rotated.astype(np.uint8)) 
    # gs_pattern = to_grayscale(r_pattern.astype(np.uint8))   

    # med_blur_gs_rotated=median_blur(gs_rotated,3)
    # rotated_lbp = lbp(med_blur_gs_rotated)
    # imshow((rotated_lbp),False,"image lbp")
    

    # # imshow(lbp(gs_pattern),title="pattern lbp")
    # # plt.show()

    # bi_rot_lbp=to_binary(rotated_lbp, adaptive=True,blockSize=11,C=0)
    # imshow(bi_rot_lbp,show=False,title="Image binary lbp ")

    # morph_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    # open_bi_lbp=closing(bi_rot_lbp,morph_kernel).astype(np.float32)
    # median_blur_open_bi_lbp = median_blur(open_bi_lbp,5)
    # imshow(median_blur_open_bi_lbp,show=False,title="Image open binary lbp ")

    # # print(rotated_lbp.shape)
    # # print(gs_pattern.shape)
    
    # bin_r_pattern = to_binary(gs_pattern.astype(np.uint8),otsu=False,thresh=240).astype(np.float32)
    # reversed_bin_pattern = 1 - bin_r_pattern
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    # reversed_bin_pattern = cv2.dilate(reversed_bin_pattern,iterations=8,kernel=kernel,borderType=cv2.BORDER_REPLICATE)
    # imshow(reversed_bin_pattern,False,title="reversed_bin_pattern pattern")

    # diff_open_bi_lbp_pattern =median_blur_open_bi_lbp - reversed_bin_pattern 
    # imshow(diff_open_bi_lbp_pattern,title="difference between bin image and pattern")

    # showCountours(img,diff_open_bi_lbp_pattern,threshold=4000)
    # c , _ =cv2.findContours(image=diff_open_bi_lbp_pattern.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    
    # diff_open_bi_lbp_pattern_cp = img.copy().astype('float')
    # imshow(diff_open_bi_lbp_pattern_cp/255,title = "img copy")
    # cracks =[]
    # for i in range(len(c)):
    #     area = cv2.contourArea(c[i])
    #     if 4000 < area:
    #         cracks.append(c[i])
    #         print("max_i = ", i,"\ncontours:",len(c))
    #         diff_open_bi_lbp_pattern_cp = cv2.drawContours(diff_open_bi_lbp_pattern_cp, [c[i]],-1, color = (0,255,0),thickness= 3)
    
    # imshow(diff_open_bi_lbp_pattern_cp/255,title="countours")



    # imshow((rotated_lbp),False)
    # imshow(lbp(gs_pattern_resized))

    # rotated_lbp_uint = np.uint8(rotated_lbp)
    # # rotated_morph = cv2.morphologyEx(rotated_lbp_uint,cv2.MORPH_OPEN,(3,3),iterations=2)
    # # imshow(rotated_morph)
    # # plt.show()

    # blur_rotated_morph = blur_freq(rotated_lbp_uint,65)
    # imshow(blur_rotated_morph,title="frourier blured image lbp")

    # diff = blur_rotated_morph - gs_pattern_resized
    # imshow(diff,title="frourier blured image lbp difference with gray scale resized pattern")
    
    # diff2 = gs_pattern_resized - blur_rotated_morph
    # for i in range(1000):
    #     diff2 = diff2 - blur_rotated_morph

    # imshow(diff2)
    # plt.show()

    # diff2 = 255 - diff2
    # imshow(diff2,title="reverse")
    # plt.show()

    # diff3 = diff2
    # # diff3 = cv2.morphologyEx(diff2,cv2.MORPH_CLOSE,(5,5),iterations=10)
    # # imshow(diff3)
    # # plt.show()
    # plt.show()

    # gs_pattern_resized2 = cv2.resize(gs_pattern, rotated_lbp.shape, interpolation = cv2.INTER_AREA).astype(float)
    # kernel = np.ones((5, 5), np.uint8)
    # gs_pattern_resized2 = cv2.erode(gs_pattern_resized2, kernel,iterations=2)
    # opening = cv2.morphologyEx(gs_pattern_resized2, cv2.MORPH_OPEN,
    #                        kernel, iterations=2) 
    # imshow(opening)
    # plt.show()

    # diff4 = opening - diff3
    # for i in range(1000):
    #     diff4 = opening - diff4
    
    # imshow(diff4)
    # plt.show()

    # pattern2 = 255 - gs_pattern_resized
    # pattern2 = cv2.morphologyEx(pattern2,cv2.MORPH_DILATE,(5,5),iterations=171)
    # imshow(pattern2)
    # plt.show()
    # for i in range(500):
    #     diff2 = diff2 - pattern2
    
    # imshow(diff2)
    # plt.show()