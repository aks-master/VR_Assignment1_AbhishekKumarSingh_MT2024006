'''
panorama creation using SIFT detector in OpenCV. Script written by Abhishek Kumar Singh.
'''

import cv2
import numpy as np
import os

def extract_keypoints_and_descriptors(image):
    sift = cv2.SIFT_create()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))
    return keypoints, descriptors, image_with_keypoints

def match_keypoints(descriptors1, descriptors2):
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def stitch_images(image1, image2):
    kp1, des1, img1_kp = extract_keypoints_and_descriptors(image1)
    kp2, des2, img2_kp = extract_keypoints_and_descriptors(image2)

    matches = match_keypoints(des1, des2)

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    height, width, channels = image1.shape
    result = cv2.warpPerspective(image2, h, (width * 2, height))
    result[0:height, 0:width] = image1

    return result, img1_kp, img2_kp

if __name__ == "__main__":
    image1 = cv2.imread('./input images/Limage1.jpg')
    image2 = cv2.imread('./input images/Rimage1.jpg')

    stitched_image, img1_kp, img2_kp = stitch_images(image1, image2)

    output_dir = "./output images/panorama"
    os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(os.path.join(output_dir, 'stitched_image.jpg'), stitched_image)
    cv2.imwrite(os.path.join(output_dir, 'image1_keypoints.jpg'), img1_kp)
    cv2.imwrite(os.path.join(output_dir, 'image2_keypoints.jpg'), img2_kp)

    cv2.namedWindow("Image 1 Keypoints", cv2.WINDOW_NORMAL)
    cv2.imshow("Image 1 Keypoints", img1_kp)
    cv2.waitKey(0)

    cv2.namedWindow("Image 2 Keypoints", cv2.WINDOW_NORMAL)
    cv2.imshow("Image 2 Keypoints", img2_kp)
    cv2.waitKey(0)

    cv2.namedWindow("Stitched Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Stitched Image", stitched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()