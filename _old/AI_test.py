from tensorflow import keras
from keras.models import load_model

#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################
#                                                     NEED TO TYPE!!!!!!!!!!!!!!!!!!
# Specify the path to the AI CNN model file. (_.h5)
model = load_model("C:/Users/junwo/Desktop/summer_research/cnn/jun_water_ditection.h5")  

#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################



from keras.preprocessing import image
import numpy as np
import cv2


#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################
#                                                     NEED TO TYPE!!!!!!!!!!!!!!!!!!

# Enter the photo path to be used to test AI.
image_path = "C:/Users/junwo/Desktop/testsample/flowA2.JPG"

#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################

img = cv2.imread(image_path)
cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("Resized_Window", 700,400)
cv2.imshow("Resized_Window", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Image loading and data preprocessing

# Resize the image to fit the model's input size. 
# In the file that creates the CNN (cnn.py), enter a number such as "target_size=(x,y)," in parentheses.
img = image.load_img(image_path, target_size=(64, 64))
img_array = image.img_to_array(img)

# Add batch dimension for Tensor
img_array = np.expand_dims(img_array, axis=0)

# This is the process of image normalization. This result is in the range 0 to 1.
img_array = img_array / 255.0 

# Images are analyzed through AI.
predictions = model.predict(img_array)

# Select results through analysis.
predicted_class = np.argmax(predictions)

# Print the analysis results. For the value of predicted_class, enter 0, 1, 2 ... in file order within 'supervisor'.
if predicted_class == 0:
    print("This is a 'flow'.")
elif predicted_class == 1:
    print("This is a 'noflow'.")
elif predicted_class == 2:
    print("This is a 'snow'.")