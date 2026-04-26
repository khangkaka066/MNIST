import torch
from PIL import Image
import numpy as np
from model import CNNmodel

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNmodel().to(device)
model.load_state_dict(torch.load("/Users/nguyenvokhang/Downloads/MNIST/mnist_cnn.pth", map_location=device))
model.eval()
# ======== KIỂM TRA THAM SỐ MÔ HÌNH ========
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Tổng số tham số: {total_params:,}")
print(f"Tham số có thể huấn luyện: {trainable_params:,}")

# In thông tin chi tiết từng layer (tên, kích thước, mean/std của trọng số conv đầu tiên)
print("\nDanh sách tham số:")
for name, param in model.named_parameters():
    print(f"{name:20} | shape: {str(list(param.shape)):15} | requires_grad: {param.requires_grad}")
    if name == "conv1.weight":
        print(f"   Mean: {param.mean().item():.4f}, Std: {param.std().item():.4f}")
# =========================================
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
predict_image("/Users/nguyenvokhang/Downloads/MNIST/mnist_6-300x295.png")