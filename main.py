constants ={
    "image_size":(1000,1000),

}

def crop(image, width=None, height=None):
    return None

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
    rotated_img=rotation_matching(image=crop_img,pattern=pattern)
    bi_img=binary_threshold(rotated_img)
    
    return None
