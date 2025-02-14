'''
Coin detection and counting using OpenCV. Script written by Abhishek Kumar Singh.
'''
import cv2
import numpy as np
import os

def detect_coins(image_path):
    # Read the input image
    image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to the grayscale image
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    # Detect edges using Canny edge detector
    edges = cv2.Canny(blurred, 30, 150)
    # Create a kernel for morphological operations
    kernel = np.ones((15, 15), np.uint8)
    # Close gaps between edges using morphological closing
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # Find contours in the closed image
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours on the original image
    cv2.drawContours(image, contours, -1, (0, 255, 0), 5)
    # Create output directory if it doesn't exist
    output_dir = "./output images/coin detection"
    os.makedirs(output_dir, exist_ok=True)
    # Save the image with detected contours
    output_path = os.path.join(output_dir, "detected_coins.jpg")
    cv2.imwrite(output_path, image)
    # Display the image with detected contours
    cv2.namedWindow("Detected Coins", cv2.WINDOW_NORMAL)
    cv2.imshow("Detected Coins", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Return the contours found
    return contours

def segment_coins(image_path, contours):
    # Read the input image
    image = cv2.imread(image_path)
    segmented_coins = []
    # Create output directory if it doesn't exist
    output_dir = "./output images/coin detection"
    os.makedirs(output_dir, exist_ok=True)
    # Loop through each contour to segment individual coins
    for i, contour in enumerate(contours):
        # Create a mask for the current contour
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [contour], -1, 255, -1)
        # Extract the coin using bitwise_and operation
        coin = cv2.bitwise_and(image, image, mask=mask)
        segmented_coins.append(coin)
        # Save the segmented coin image
        output_path = os.path.join(output_dir, f"segmented_coin_{i+1}.jpg")
        cv2.imwrite(output_path, coin)
        # Display the segmented coin image
        cv2.namedWindow(f"Segmented Coin {i+1}", cv2.WINDOW_NORMAL)
        cv2.imshow(f"Segmented Coin {i+1}", coin)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # Return the list of segmented coins
    return segmented_coins

def count_coins(contours):
    # Return the number of contours found
    return len(contours)

if __name__ == "__main__":
    # Path to the input image
    image_path = "./input images/coins.jpg"
    # Detect coins in the image
    contours = detect_coins(image_path)
    # Segment the detected coins
    segmented_coins = segment_coins(image_path, contours)
    # Count the total number of coins detected
    total_coins = count_coins(contours)
    print(f"Total number of coins detected: {total_coins}")
    # Save the total count of coins to a text file
    output_dir = "./output images/coin detection"
    count_file_path = os.path.join(output_dir, "total_coins.txt")
    with open(count_file_path, "w") as f:
        f.write(f"Total number of coins detected: {total_coins}\n")