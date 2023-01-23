"""
For creating auxiliary functions
"""
import cv2 
import scipy.signal as sg
import scipy.stats as st
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern
import json
import math

constants ={
    "image_size":(1200,1200),
    "image_folder":"images/",
    "kernel_size":9,
    "resized_dim":1600
}
def to_grayscale(im):
    if im.shape[2] >1:
        # output = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        output = cv2.cvtColor(im, cv2.COLOR_BGR2HSV_FULL)[:,:,2]
    else:
        output = im
    return output

def opening(image,kernel):
    
    dialect=cv2.dilate(image,kernel=kernel,borderType=cv2.BORDER_REPLICATE)
    erode=cv2.erode(dialect,kernel=kernel,iterations=1,borderType=cv2.BORDER_REPLICATE)
    return(erode)
def closing(image,kernel):
    
    erode=cv2.erode(image,kernel=kernel,iterations=1,borderType=cv2.BORDER_REPLICATE)    
    dialect=cv2.dilate(erode,kernel=kernel,borderType=cv2.BORDER_REPLICATE)
    return(dialect)

def crop_out(im, vertices, size=None):
    if size is None :
        width, height = constants["resized_dim"] , constants["resized_dim"]
    else:
        width, height = size
    target = np.array([[0,0],[0,height],[width,height],[width,0]])
    transform = cv2.getPerspectiveTransform(vertices.astype(np.float32), target.astype(np.float32))  # get the top or bird eye view effect
    return cv2.warpPerspective(src=im,M= transform,dsize= (width, height)),transform


# def make_img_from_label_vertices(base_img , vertices):
#     height , width , depth = base_img.shape
#     img = np.zeros((height,width),dtype=np.uint8)
#     for shape in vertices:
#         for point in shape:
#             img[int(point[0]),int(point[1])] = 255
#     return img

def load_vertices_from_json(json_file):
    vertices = []
    json_file = constants['image_folder']+"/"+json_file
    f = open(json_file)
    data = json.load(f)
    shapes = data['shapes']
    for shape in shapes:
        points = shape['points']
        vertices.append(points)
    return vertices


def transform_vertices(json_file,transform):
    print("transform_vertices:")
    vertices = load_vertices_from_json(json_file)
    transformed_vertices = []
    for points in vertices :
        shape = []
        for point in points:
            point = np.float32(np.array([[point]])) 
            shape.append(cv2.perspectiveTransform(np.array(point), transform))
        transformed_vertices.append(shape)
    return transformed_vertices


def rotate_point(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    print(point)
    px, py = (point[0][0][0],point[0][0][1])

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    (point[0][0][0],point[0][0][1]) = (px, py)

    return point

def rotate_vertices(angle,origin_point,vertices):
    for points in vertices:
        for point in points:
            point = rotate_point(origin_point,point,angle)
    return vertices

def transform_labels(json_file ,transform ,angle):
    transformed_vertices = transform_vertices(json_file,transform)
    rotated_transformed_vertices = rotate_vertices(angle,origin_point=(constants["resized_dim"]/2,constants["resized_dim"]/2),vertices=transformed_vertices)
    return rotated_transformed_vertices

def show_transfered_labels(image,vertices):
    print("show_vertices:")
    image_cp = image.copy()
    for points in vertices:
        print(points)
        shape = []
        for point in points:
            p = [[int(point[0][0][0]),(point[0][0][1])]]
            #p = np.array([[int(point[0][0][0]),(point[0][0][1])]], dtype=np.int32)
            shape.append(p)
        shape = np.array(shape,dtype=np.int32)
        print(f"shape : {shape}")
        image_cp = cv2.drawContours(image_cp, shape,-1, color = (10,250,0),thickness= 5)
    return image_cp

def to_edges(im,lower=40,upper=150):
    """
    Return edges using Canny edge detector
    :params: lower: Lower threshold for edges
    :params: upper: upper threshold for edges
    """
    im = im.copy()
    # sharpen_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    # im=cv2.filter2D(im,1,sharpen_kernel,borderType=cv2.BORDER_REFLECT)
    edge = cv2.Canny(im.astype(np.uint8),lower,upper)
    return edge

def to_binary(im,adaptive=False,otsu=True,thresh=200,max_value=1,blockSize=7,C=4):
    im = im.copy().astype(np.uint8)

    if adaptive is False:
        if otsu:
            t,bi = cv2.threshold(im,thresh,max_value,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        else:
            t,bi = cv2.threshold(im,thresh,max_value,cv2.THRESH_BINARY)
        print("threshold = ",t)
    else:        
        bi=cv2.adaptiveThreshold(im,max_value,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,blockSize=blockSize,C=C)

    return bi

def gkern(kernlen=constants["kernel_size"], nsig=4):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

def blur(im,kernel_size=constants["kernel_size"]):
    """Returns a 2D Gaussian blured image"""
    
    kernel = gkern(kernel_size)
    b = sg.correlate(im,kernel,mode='same')
    return b

def median_blur(im,ksize=constants["kernel_size"]):
    """Returns a 2D median blured image"""
    mb=cv2.medianBlur(im,ksize=ksize)
    return mb

def find_vertices(im):
    """
    find biggest contour in the image
    """
    c , _ =cv2.findContours(image=im, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    
    max_i = 0
    max_a = 0
    for i in range(len(c)):
        area = cv2.contourArea(c[i])

        if max_a < area:
            max_a = area
            max_i = i
    print("max_i = ", max_i,"\ncontours:",len(c))
    curve=cv2.approxPolyDP(c[max_i],epsilon=20,closed=True)

    return curve


def lbp(image):
    METHOD = 'ror'
    lbp_image = local_binary_pattern(image, 8, 1, METHOD)
    return lbp_image

def imshow(im,show=True,title="figure"):
    plt.figure()
    width, height, *channels = im.shape
    if channels:
        # By default, OpenCV tends to work with images in the BGR format.
        # This is due to some outdated practices, but it has been left in the library.
        # We can iterate the channels in reverse order to get an RGB image.
        plt.imshow(im[:,:,::-1])
    else:
        plt.imshow(im, cmap='gray')
    plt.axis('off')
    plt.title(title)
    if show:
        plt.show()

def blur_freq(im,threshold):
    # bure image based on frequency 
    result = im.copy()
    denoised = im.copy()
    denoised = np.fft.fft2(denoised)
    denoised = np.fft.fftshift(denoised)

    row,col = im.shape[0],im.shape[1]
    shape = (row , col)
    center = (row/2 , col/2)

    freq = np.zeros(shape)
    # above_threshold_counter = 0
    # below_threshold_counter = 0

    for x in range(row):
        for y in range(col):
            dist = np.sqrt((x - center[0])**2 + (y-center[1])**2)
            if dist < threshold:
                freq[x][y] = 1
            # if dist > 150:
            #     above_threshold_counter += 1
            # else:
            #     below_threshold_counter += 1
            
    
    denoised = denoised * freq
    denoised = np.fft.ifftshift(denoised)
    denoised = np.fft.ifft2(denoised)

    for x in range(row):
        for y in range(col):
            result[x][y] = float(denoised[x][y])

    # print(above_threshold_counter)
    # print(below_threshold_counter)

    return result

def sliding_window(image,stride_width,stride_height,window_size=[50,50]):
    width = image.shape[1]
    height = image.shape[0]

    windows =  []
    for i in range(0,width,stride_width):
        for j in range(0,height,stride_height):
            if i + window_size[0] < width:
                if j + window_size[1] < height:
                    windows.append(image[j:j + window_size[1], i:i+window_size[0]])

    return windows

def showCountours(base_image , display_image, threshold = 4000):
    """
    finds contours on 
    """
    c , _ =cv2.findContours(image=display_image.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    
    diff_open_bi_lbp_pattern_cp = base_image.copy().astype('float')
    # imshow(diff_open_bi_lbp_pattern_cp/255,title = "img copy")
    cracks =[]
    for i in range(len(c)):
        ci=c[i]
        # ci=cv2.approxPolyDP(ci,epsilon=10,closed=True) # reduce the number of points in contour

        area = cv2.contourArea(ci)
        if threshold < area:
            cracks.append(ci)
            print("max_i = ", i,"\ncontours:",len(c))
            diff_open_bi_lbp_pattern_cp = cv2.drawContours(diff_open_bi_lbp_pattern_cp, [c[i]],-1, color = (10,250,0),thickness= 2)
    imshow(diff_open_bi_lbp_pattern_cp/255,title="countours")
    return (diff_open_bi_lbp_pattern_cp/255)