import cv2
import numpy as np
from utils.transform import corners_to_img
from utils.transform import projectToImage

def drawBox3D(img, corners):

    # img = np.copy(img)
    corners = corners.astype(np.int32)

    cv2.line(img, (corners[0,0], corners[1,0]), (corners[0,1], corners[1,1]), thickness=2, color=(0, 255, 255))
    cv2.line(img, (corners[0,1], corners[1,1]), (corners[0,2], corners[1,2]), thickness=2, color=(0, 255, 255))
    cv2.line(img, (corners[0,2], corners[1,2]), (corners[0,3], corners[1,3]), thickness=2, color=(0, 255, 255))
    cv2.line(img, (corners[0,3], corners[1,3]), (corners[0,0], corners[1,0]), thickness=2, color=(0, 255, 255))

    cv2.line(img, (corners[0,4], corners[1,4]), (corners[0,5], corners[1,5]), thickness=2, color=(0, 255, 255))
    cv2.line(img, (corners[0,5], corners[1,5]), (corners[0,6], corners[1,6]), thickness=2, color=(0, 255, 255))
    cv2.line(img, (corners[0,6], corners[1,6]), (corners[0,7], corners[1,7]), thickness=2, color=(0, 255, 255))
    cv2.line(img, (corners[0,7], corners[1,7]), (corners[0,4], corners[1,4]), thickness=2, color=(0, 255, 255))

    cv2.line(img, (corners[0,0], corners[1,0]), (corners[0,4], corners[1,4]), thickness=2, color=(0, 255, 255))
    cv2.line(img, (corners[0,1], corners[1,1]), (corners[0,5], corners[1,5]), thickness=2, color=(0, 255, 255))
    cv2.line(img, (corners[0,2], corners[1,2]), (corners[0,6], corners[1,6]), thickness=2, color=(0, 255, 255))
    cv2.line(img, (corners[0,3], corners[1,3]), (corners[0,7], corners[1,7]), thickness=2, color=(0, 255, 255))

    return img


def show_lidar_corners(test_image, lidar_corners, calib):
    test = np.copy(test_image)
    for i in range(lidar_corners.shape[0]):
        img_corners = corners_to_img(lidar_corners[i], calib[3], calib[2], calib[0])
        img = drawBox3D(test, img_corners/img_corners[2,:])
    return img

def show_cam_corners(test_image, cam_corners, calib):
    test = np.copy(test_image)
    for i in range(cam_corners.shape[0]):
        if cam_corners[i].shape[0] == 24:
            cam_corners_i = cam_corners[i].reshape((3, 8))
        img_corners = projectToImage(cam_corners_i, calib[0])
        img = drawBox3D(test, img_corners)
    return img

def show_image_boxes(test_image, img_boxes):
    test = np.copy(test_image)
    num = len(img_boxes)
    for n in range(num):
        x1,y1,x2,y2 = img_boxes[n]
        cv2.rectangle(test,(x1,y1), (x2,y2), (255,255,0), 2)
    return test
