import cv2
import hcd

filename = ""
img = cv2.imread(filename)
result = hcd.harris_corner_detection(img)

cv2.imshow('Result', result)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
