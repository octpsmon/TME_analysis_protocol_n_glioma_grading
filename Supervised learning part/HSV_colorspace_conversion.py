import cv2
import os
import glob

def convert_to_hsv_and_save(input_image_path, output_image_path):
    # Read the image
    image = cv2.imread(input_image_path)

    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Save the converted image
    cv2.imwrite(output_image_path, hsv_image)

def process_images(input_dir, output_dir):
    # Check if the output directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all class folders
    for class_folder in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_folder)
        output_class_path = os.path.join(output_dir, class_folder)

        # Check if class folder in output directory exists, if not, create it
        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)

        # Process all images in the class folder
        for image_file in glob.glob(os.path.join(class_path, '*.*')):  # Adjust the pattern as needed
            output_image_path = os.path.join(output_class_path, os.path.basename(image_file))
            convert_to_hsv_and_save(image_file, output_image_path)

# Define the input and output directories
input_directory = '/home/monika/HLA-TMA-cropped/cropped'
output_directory = '/home/monika/HLA_hsv'

# Process the images
process_images(input_directory, output_directory)

print("HSV color space conversion applied to all images.")
