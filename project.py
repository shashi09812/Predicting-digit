import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image, ImageOps
import numpy as np

# Load the dataset
(x_train,y_train), (x_test,y_test) = mnist.load_data()

# Normalizing the data
x_train, x_test = x_train/255.0, x_test/255.0

# Add channel dimension for CNN
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Display the first image in the dataset
plt.imshow(x_train[0].reshape(28, 28), cmap='gray')
plt.title(f"Training example - Label: {y_train[0]}")
plt.show()

# Build an improved model with CNN layers
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Function to preprocess the image
def preprocess_image(image_path):
    # Load and convert to grayscale
    img = Image.open(image_path).convert('L')
    
    # Invert if needed (MNIST has white digits on black background)
    img = ImageOps.invert(img)
    
    # Resize to 28x28
    img = img.resize((28, 28))
    
    # Convert to array and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Threshold to help with noise reduction
    img_array = (img_array > 0.3).astype(np.float32)
    
    # Add batch and channel dimensions
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array

# Path to the handwritten digit image
image_path = 'digit.png'

try:
    # Preprocess and predict
    new_image = preprocess_image(image_path)
    
    # Display the preprocessed image
    plt.figure(figsize=(3, 3))
    plt.imshow(new_image.reshape(28, 28), cmap='gray')
    plt.title("Preprocessed Input Image")
    plt.show()
    
    # Make prediction
    prediction = model.predict(new_image)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    # Display prediction with confidence
    print(f"Predicted Digit: {predicted_digit} (Confidence: {confidence:.2f}%)")
    
    # Show all prediction probabilities
    plt.figure(figsize=(8, 4))
    plt.bar(range(10), prediction[0] * 100)
    plt.xticks(range(10))
    plt.xlabel('Digit')
    plt.ylabel('Probability (%)')
    plt.title('Prediction Probabilities')
    plt.show()
    
except Exception as e:
    print(f"Error processing the image: {e}")
    print("Please check if the path to your image is correct.")