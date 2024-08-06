import tensorflow as tf

# References
# https://www.tensorflow.org/tutorials/images/classification
# https://campus.datacamp.com/courses/introduction-to-deep-learning-with-keras/improving-your-model-performance
# https://github.com/jeffheaton/t81_558_deep_learning/blob/b59a6ca8334246a7b5358c81b9c7b1849bb45371/t81_558_class_06_2_cnn.ipynb

# TODO verify validation vs test
# TODO try larger image size
# TODO setup labeling of true and false classification results: see Jun's code

#################
# User Parameters
################

# Path to the ‘supervisor’ file.
data_dir = r"C:\Users\josep\OneDrive - UCB-O365\Students\_shares\Lee HUB\junresearch\DeeplearningCNN_flow_detection\supervisor"

# Path to the output file
output_file = r"C:\Users\josep\Documents\GitHub\MLtests\flow_detection\model_target680_batch16.keras"

#################
# Model creation and testing
################

edge_size = 680

# Create ImageDataGenerator to load and preprocess image data
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.0,  # image normalization
    validation_split=0.2  # Set verification data split ratio
)

# Load data for training
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(edge_size, edge_size),  # Image resizing (Jun had 64x64)
    batch_size=16,          # Set batch size (Jun had 64)
    class_mode='categorical',  # Multi-class classification
    subset='training'          # training data
)

# Load data for validation
validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(edge_size, edge_size),  # Jun had 64x64
    batch_size=16, #Jun had 64
    class_mode='categorical',
    subset='validation'  # validation data
)

#Find an appropriate avtivation instead of 'relu' to reduce loss
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(edge_size, edge_size, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 classes, this number is the number of files in 'supervisor'
])


# Compile the model with the desired learning rate

# Jun version
#new_learning_rate = 0.0001
#optimizer = tf.keras.optimizers.Adam(learning_rate=new_learning_rate)

# new version
optimizer = tf.keras.optimizers.Adam(gradient_accumulation_steps=2)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Jun: Set class weights. By doing this, the difference in the number of photos in each file is overcome.
#class_weight = {"flow": 0.50, "noflow": 0.20, "snow": 0.30}

# adding early stopping
# (https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_03_4_early_stop.ipynb)
monitor = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                           min_delta=1e-3,
                                           patience=5,
                                           mode='auto',
                                           restore_best_weights=True)

# model training
history = model.fit(
        train_generator,
        epochs=15,  # Set epochs count
        validation_data=validation_generator,
        callbacks=[monitor]
    )

# Model evaluation and self-diagnosis
test_loss, test_accuracy = model.evaluate(validation_generator)
print(f'Test accuracy: {test_accuracy}')

# Image classification prediction
predictions = model.predict(validation_generator)

# Specify the path and file name to save the model C:/path/file_name.h5
model.save(output_file)
