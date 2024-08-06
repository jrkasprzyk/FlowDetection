from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LeakyReLU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# Set data directory path
data_dir = "C:/Users/junwo/Desktop/summer_research/cnn/supervisor"

# Image data loading and preprocessing settings
datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Image normalization
    validation_split=0.2,  # Set verification data split ratio
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load training data
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(64, 64),  # Resize image
    batch_size=64,  # Set batch size
    class_mode='categorical',  # Multi-class classification
    subset='training'  # Set as training data
)

# Load verification data
validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(64, 64),
    batch_size=64,
    class_mode='categorical',
    subset='validation' # Set as verification data
)

# model definition
model = Sequential([
    Conv2D(32, (3, 3), activation=LeakyReLU(alpha=0.1), input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation=LeakyReLU(alpha=0.1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation=LeakyReLU(alpha=0.1)),
    Flatten(),
    Dense(64, activation=LeakyReLU(alpha=0.1)),
    Dense(3, activation='softmax')  # Classified into 3 classes
])

# Compile model
new_learning_rate = 0.0001
optimizer = Adam(learning_rate=new_learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Early termination settings
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

# Model training
history = model.fit(
    train_generator,
    epochs=50,  # Increased number of epochs
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

# Model evaluation
test_loss, test_accuracy = model.evaluate(validation_generator)
print(f'Test accuracy: {test_accuracy}')

# Image classification prediction
predictions = model.predict(validation_generator)

# Save model
model.save("C:/Users/junwo/Desktop/summer_research/cnn/jun_water_detection.h5")