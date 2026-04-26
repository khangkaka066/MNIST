import torch
from PIL import Image
import numpy as np
from model import CNNmodel

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNmodel().to(device)
model.load_state_dict(torch.load("/kaggle/working/mnist_cnn.pth", map_location=device))
model.eval()

def predict_image(image_path):
    img = Image.open(image_path).convert('L')  # grayscale
    img = img.resize((28, 28))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = (img_array - 0.1307) / 0.3081  # normalize
    tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(dim=1).item()
    print(f"Predicted digit: {pred}")

# Thử với ảnh bạn có
predict_image("/kaggle/input/some_digit.png")