import os
import shutil
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image

#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################
#                                                     NEED TO TYPE!!!!!!!!!!!!!!!!!!

# Select the file path containing the photos to be separated through AI.
input_image_dir = "C:/Users/junwo/Desktop/test"

#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################



#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################
#                                                     NEED TO TYPE!!!!!!!!!!!!!!!!!!

# File path where photos categorized through AI will be saved
output_dir = "C:/Users/junwo/Desktop/cnntestresult"

#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################



#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################
#                                                     NEED TO TYPE!!!!!!!!!!!!!!!!!!

# Enter the path of the AI model that will be used to analyze this photo.
model_path = "C:/Users/junwo/Desktop/summer_research/cnn/jun_waternosnow_ditection.h5"

#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################


# Category definition: Specify the categories into which photos will be separated. 
# It is recommended to use the file name in 'supervisor' used for AI learning. 
# EX) If there are ‘flow’, ‘noflow’, and ‘snow’ in the supervisor, just write them down.
categories = ['flow', 'noflow']

# The CNN model run.
model = tf.keras.models.load_model(model_path)

# Create result directory according to path
os.makedirs(output_dir, exist_ok=True)

# Import image file list
image_files = os.listdir(input_image_dir)

# Classify images through AI and create a folder to store the results
for filename in image_files:
    image_path = os.path.join(input_image_dir, filename)
    
    # Resize the image to fit the model's input size. 
    # In the file that creates the CNN (cnn.py), enter a number such as "target_size=(x,y)," in parentheses.
    img = image.load_img(image_path, target_size=(64, 64)) 
    
    img = image.img_to_array(img)
    img = tf.expand_dims(img, axis=0)
    img = img / 255.0

    # Predict with model
    predictions = model.predict(img)
    predicted_class = categories[predictions.argmax()]

    # Create directory by category
    output_category_dir = os.path.join(output_dir, predicted_class)
    os.makedirs(output_category_dir, exist_ok=True)

    # Copy the image to the corresponding category folder
    shutil.copy(image_path, os.path.join(output_category_dir, filename))

print("done")