import os
import cv2 as cv

# ---------- READ INPUT IMAGES FROM FOLDER AS GRAYSCALE ----------
def read_from_path(path):
    images = []
    for filename in os.listdir(path):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            img_path = os.path.join(path, filename)
            img = cv.imread(img_path)
            if img is not None:
                images.append(img)
    return images

# Specify the path to the folder containing images
path = os.getcwd() + "/input"

# Read images from the folder
images = read_from_path(path)

# Show images 
def show(img_list):
    i = 1
    for img in img_list: 
        cv.imshow('Image ' + str(i), img)
        i = i + 1

# Convert to binary 
def cvt2G(img_list): 
    images_grayscale = []
    for img in img_list: 
        images_grayscale.append(cv.cvtColor(img, cv.COLOR_RGB2GRAY))
    return images_grayscale

# ---------- CONVERT TO BINARY ----------
def cvt2B(img_list):
    images_binary = [] 
    for img in img_list: 
        ret, th = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        images_binary.append(th)
    return images_binary 

# ---------- PRE PROCESSING ----------
def preProcessing(img_list): 
    images_pre = [] 
    for img in img_list:
        # Blur 
        blur = cv.medianBlur(img, 7)


        # Append to the list 
        images_pre.append(blur) 

    return images_pre


# Display images 
show(images)
images_grayscale = cvt2G(images)
show(cvt2B(images_grayscale))
show(cvt2B(images_grayscale))

cv.waitKey()
cv.destroyAllWindows() 

