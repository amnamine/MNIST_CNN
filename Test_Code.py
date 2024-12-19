import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

# Define the CNN model (should match the original definition)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained model
model_path = "mnist_cnn_model.pt"
model = SimpleCNN()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()  # Set model to evaluation mode

# Transformation for the image
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure it's a single-channel image
    transforms.Resize((28, 28)),  # Resize to 28x28
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize to match training data
])

# Predict function
def predict_digit(image_path):
    try:
        image = Image.open(image_path)
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        return predicted.item()
    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed: {e}")
        return None

# GUI for digit prediction
class DigitPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Predictor")
        self.root.geometry("400x500")
        self.image_path = None

        # Add label
        self.label = tk.Label(root, text="MNIST Digit Predictor", font=("Arial", 18), pady=10)
        self.label.pack()

        # Add canvas for image preview
        self.canvas = tk.Canvas(root, width=200, height=200, bg="lightgray")
        self.canvas.pack(pady=10)

        # Add button to load image
        self.load_btn = tk.Button(root, text="Load Image", command=self.load_image, font=("Arial", 14))
        self.load_btn.pack(pady=5)

        # Add button to predict digit
        self.predict_btn = tk.Button(root, text="Predict", command=self.predict_digit, font=("Arial", 14))
        self.predict_btn.pack(pady=5)

        # Add label to show the prediction result
        self.result_label = tk.Label(root, text="", font=("Arial", 16), pady=10)
        self.result_label.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.image_path = file_path
            image = Image.open(file_path)
            image.thumbnail((200, 200))
            self.img = ImageTk.PhotoImage(image)
            self.canvas.create_image(100, 100, image=self.img)
            self.result_label.config(text="")

    def predict_digit(self):
        if not self.image_path:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        digit = predict_digit(self.image_path)
        if digit is not None:
            self.result_label.config(text=f"Predicted Digit: {digit}")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitPredictorApp(root)
    root.mainloop()
