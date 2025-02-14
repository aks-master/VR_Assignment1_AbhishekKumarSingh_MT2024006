'''
Panorama creation using SIFT detector in OpenCV. Script written by Abhishek Kumar Singh.
'''

import cv2
import numpy as np
import os

def extract_keypoints_and_descriptors(image):
    # Create SIFT detector object
    sift = cv2.SIFT_create()
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    # Draw keypoints on the image
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))
    return keypoints, descriptors, image_with_keypoints

def match_keypoints(descriptors1, descriptors2):
    # Create BFMatcher object with L2 norm and cross-check enabled
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # Match descriptors
    matches = matcher.match(descriptors1, descriptors2)
    # Sort matches based on distance (lower distance is better)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def stitch_images(image1, image2):
    # Extract keypoints and descriptors from both images
    kp1, des1, img1_kp = extract_keypoints_and_descriptors(image1)
    kp2, des2, img2_kp = extract_keypoints_and_descriptors(image2)

    # Match keypoints between the two images
    matches = match_keypoints(des1, des2)

    # Initialize arrays to hold matched keypoints
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    # Extract location of good matches
    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    # Find homography matrix using RANSAC
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Get dimensions of the first image
    height, width, channels = image1.shape
    # Warp the second image to align with the first image
    result = cv2.warpPerspective(image2, h, (width * 2, height))
    # Place the first image in the result
    result[0:height, 0:width] = image1

    return result, img1_kp, img2_kp

if __name__ == "__main__":
    # Read input images
    image1 = cv2.imread('./input images/Limage1.jpg')
    image2 = cv2.imread('./input images/Rimage1.jpg')

    # Stitch the images together
    stitched_image, img1_kp, img2_kp = stitch_images(image1, image2)

    # Create output directory if it doesn't exist
    output_dir = "./output images/panorama"
    os.makedirs(output_dir, exist_ok=True)

    # Save the stitched image and images with keypoints
    cv2.imwrite(os.path.join(output_dir, 'stitched_image.jpg'), stitched_image)
    cv2.imwrite(os.path.join(output_dir, 'image1_keypoints.jpg'), img1_kp)
    cv2.imwrite(os.path.join(output_dir, 'image2_keypoints.jpg'), img2_kp)

    # Display images with keypoints and the stitched image

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