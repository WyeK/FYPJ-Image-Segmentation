import cv2
import pixellib
from pixellib.instance import instance_segmentation


segment_image = instance_segmentation()
segment_image.load_model("mask_rcnn_coco.h5")
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while camera.isOpened():
    res, frame = camera.read()
    segmask, output = segment_image.segmentFrame(frame, show_bboxes=True)
    image = output
    print(str(segmask["rois"]))
    cv2.imshow("Image Segmentation", image)
    cv2.waitKey(1)

camera.release()
cv2.destroyAllWindows()
