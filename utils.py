"""
For creating auxiliary functions
"""
import cv2 
import scipy.signal as sg
import scipy.stats as st
import numpy as np

constants ={
    "image_size":(1200,1200),
    "kernel_size":9,
}
def to_grayscale(im):
    if im.shape[2] >1:
        output = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        output = im
    return output

def to_edges(im,lower=100,upper=350):
    """
    Return edges using Canny edge detector
    :params: lower: Lower threshold for edges
    :params: upper: upper threshold for edges
    """
    edge = cv2.Canny(im.astype(np.uint8),lower,upper)
    return edge

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
    c , _ =cv2.findContours(image=im, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    
    max_i = 0
    max_a = 0
    for i in range(len(c)):
        area = cv2.contourArea(c[i])

        if max_a < area:
            max_a = area
            max_i = i
    
    curve=cv2.approxPolyDP(c[max_i],epsilon=20,closed=True)

    return curve
