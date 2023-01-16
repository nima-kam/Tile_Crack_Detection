"""
For creating auxiliary functions
"""
import cv2 
import scipy.signal as sg
import scipy.stats as st
import numpy as np
from matplotlib import pyplot as plt

constants ={
    "image_size":(1200,1200),
    "kernel_size":9,
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

def crop_out(im, vertices, size=None):
    if size is None:
        width, height = im.shape[1]-1 , im.shape[0]-1
    else:
        width, height = size
    target = np.array([[0,0],[0,height],[width,height],[width,0]])
    transform = cv2.getPerspectiveTransform(vertices.astype(np.float32), target.astype(np.float32))  # get the top or bird eye view effect
    return cv2.warpPerspective(src=im,M= transform,dsize= (width, height))

def to_edges(im,lower=40,upper=150):
    """
    Return edges using Canny edge detector
    :params: lower: Lower threshold for edges
    :params: upper: upper threshold for edges
    """
    im = im.copy()
    sharpen_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    im=cv2.filter2D(im,1,sharpen_kernel,borderType=cv2.BORDER_REFLECT)
    edge = cv2.Canny(im.astype(np.uint8),lower,upper)
    return edge

def to_binary(im,adaptive=False,otsu=True,thresh=200):
    im = im.copy().astype(np.uint8)

    if adaptive is False:
        if otsu:
            t,bi = cv2.threshold(im,thresh,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        else:
            t,bi = cv2.threshold(im,thresh,1,cv2.THRESH_BINARY)
        print("threshold = ",t)
    else:        
        bi=cv2.adaptiveThreshold(im,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,17,10)

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

def imshow(im,show=True):
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
    if show:
        plt.show()