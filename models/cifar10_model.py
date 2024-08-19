import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image

# Load the model
model = load_model("cifar10_model.h5")

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def plot_img(img):
    plt.figure(figsize=(2, 2))
    plt.imshow(img)
    plt.show()

def preprocess_image(path):
    img = Image.open(path).convert('RGB')
    # Resize the image to match the input shape of the model (32x32 pixels)
    img = img.resize((32, 32))
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Plot the image
    plot_img(img)
    # Normalize the image array
    img_array = img_array / 255.0
    # Expand the dimensions to match the model's expected input shape
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(path):
    img_array = preprocess_image(path)
    # Make predictions
    predictions = model.predict(img_array)
    # Get the index of the highest probability class
    predicted_class = np.argmax(predictions, axis=1)[0]
    # Return the class name
    return class_names[predicted_class]

img_path = "test3.png"  # change here

print(predict(img_path))
