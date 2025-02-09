'''
coin detection and counting using OpenCV. Script written by Abhishek Kumar Singh.
'''
import cv2
import numpy as np
import os

def detect_coins(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    edges = cv2.Canny(blurred, 30, 150)
    kernel = np.ones((15, 15), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 5)
    output_dir = "./output images/coin detection"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "detected_coins.jpg")
    cv2.imwrite(output_path, image)
    cv2.namedWindow("Detected Coins", cv2.WINDOW_NORMAL)
    cv2.imshow("Detected Coins", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return contours

def segment_coins(image_path, contours):
    image = cv2.imread(image_path)
    segmented_coins = []
    output_dir = "./output images/coin detection"
    os.makedirs(output_dir, exist_ok=True)
    for i, contour in enumerate(contours):
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [contour], -1, 255, -1)
        coin = cv2.bitwise_and(image, image, mask=mask)
        segmented_coins.append(coin)
        output_path = os.path.join(output_dir, f"segmented_coin_{i+1}.jpg")
        cv2.imwrite(output_path, coin)
        cv2.namedWindow(f"Segmented Coin {i+1}", cv2.WINDOW_NORMAL)
        cv2.imshow(f"Segmented Coin {i+1}", coin)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return segmented_coins

def count_coins(contours):
    return len(contours)

if __name__ == "__main__":
    image_path = "./input images/coins.jpg"
    contours = detect_coins(image_path)
    segmented_coins = segment_coins(image_path, contours)
    total_coins = count_coins(contours)
    print(f"Total number of coins detected: {total_coins}")
    output_dir = "./output images/coin detection"
    count_file_path = os.path.join(output_dir, "total_coins.txt")
    with open(count_file_path, "w") as f:
        f.write(f"Total number of coins detected: {total_coins}\n")