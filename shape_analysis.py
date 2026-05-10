import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image from the images folder
image_path = 'images/Shapes1.bmp'  # Updated to BMP format as per the available file
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found. Please check the path.")
else:
    print("Image loaded successfully.")
    # Placeholder for morphological operations and image difference analysis
    # Add your analysis code here