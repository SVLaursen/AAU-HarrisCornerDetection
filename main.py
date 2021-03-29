import cv2 as cv
import hcd

# READ IMG FILE
filename = "./res/imgs/shapes.png"
img = cv.imread(filename)

# USE HARRIS CORNER DETECTION
result = hcd.harris_corner_detection(img, 0.04, 0.01, 0.001)

# SHOW RESULTS
cv.imshow('Original', img)
cv.imshow('Result', result)

# KEEP THE IMAGES AROUND TILL TOLD TO SHUTDOWN
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()
