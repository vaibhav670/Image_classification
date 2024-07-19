import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tensorflow.keras.models import model_from_json
import numpy as np
from PIL import Image, ImageTk

# Load the trained model
json_file = open("/Users/rajvansh3001icloud.com/Downloads/CIFAR-10_Image_Classification-main/model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("/Users/rajvansh3001icloud.com/Downloads/CIFAR-10_Image_Classification-main/model.h5")
# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        image = image.resize((200, 200))
        photo = ImageTk.PhotoImage(image)

        # Update the image on the GU
        image_label.config(image=photo)
        image_label.image = photo
        classify_button.config(state=tk.NORMAL)

        # Convert the image to the format required by the model
        global loaded_image
        loaded_image = image.convert('RGB')
        
def classify_image():
    global loaded_image
    try:
        # Preprocess the image
        loaded_image = loaded_image.resize((32, 32))
        img_array = np.array(loaded_image)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make the prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        # Show the prediction result
        messagebox.showinfo("Prediction", f"The predicted class is: {class_names[predicted_class]}")
    except Exception as e:
        messagebox.showerror("Error", f"Error occurred: {e}")

# Create the main window
root = tk.Tk()
root.title("CIFAR-10 Image Classification")

# Image display area
image_label = tk.Label(root)
image_label.pack()

# Load Image button
load_button = tk.Button(root, text="Load Image", command=load_image)
load_button.pack()

# Classify button
classify_button = tk.Button(root, text="Classify Image", command=classify_image, state=tk.DISABLED)
classify_button.pack()

# Run the Tkinter event loop
root.mainloop()
