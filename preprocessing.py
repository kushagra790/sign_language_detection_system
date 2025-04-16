# without using any gaussian blur and gray for preprocessing 
import os
import cv2
import random

if not os.path.exists("data2"):
    os.makedirs("data2")
if not os.path.exists("data2/train"):
    os.makedirs("data2/train")
if not os.path.exists("data2/test"):
    os.makedirs("data2/test")

train_path = "SignImage48x48"

# Percentage of images for training
train_percentage = 0.75

for (dirpath, dirnames, filenames) in os.walk(train_path):
    for dirname in dirnames:
        print(dirname)
        for (direcpath, direcnames, files) in os.walk(train_path + "/" + dirname):
            if not os.path.exists("data2/train/" + dirname):
                os.makedirs("data2/train/" + dirname)
            if not os.path.exists("data2/test/" + dirname):
                os.makedirs("data2/test/" + dirname)

            # Randomly shuffle the list of files
            random.shuffle(files)

            # Determine the number of images for training and testing
            num_train = int(len(files) * train_percentage)
            num_test = len(files) - num_train

            for i, filename in enumerate(files):
                actual_path = train_path + "/" + dirname + "/" + filename
                output_folder = "train" if i < num_train else "test"
                output_filepath = "data2/" + output_folder + "/" + dirname + "/" + filename

                # Attempt to read the image
                img = cv2.imread(actual_path)

                # Check if the image was read successfully
                if img is None:
                    print(f"Error: Unable to read image '{actual_path}'")
                    continue  # Skip to the next image

                # Write the image to the output file
                cv2.imwrite(output_filepath, img)

