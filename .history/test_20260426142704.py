import torch
from PIL import Image
import numpy as np
from model import CNNmodel
import cv2

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

# # In thông tin chi tiết từng layer (tên, kích thước, mean/std của trọng số conv đầu tiên)
# print("\nDanh sách tham số:")
# for name, param in model.named_parameters():
#     print(f"{name:20} | shape: {str(list(param.shape)):15} | requires_grad: {param.requires_grad}")
#     if name == "conv1.weight":
#         print(f"   Mean: {param.mean().item():.4f}, Std: {param.std().item():.4f}")


def preprocess_image(image_path, invert=True, mean=0.1307, std=0.3081):
    """
    Tiền xử lý ảnh để đưa vào mô hình MNIST.
    - invert: nếu ảnh có nền sáng, chữ tối -> True (đảo thành nền tối, chữ sáng)
    """
    # Đọc ảnh grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

    # Đảo màu nếu cần (mặc định ảnh thực tế là nền trắng chữ đen)
    if invert:
        img = 255 - img

    # Làm mờ nhẹ để giảm nhiễu
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Tìm bounding box của chữ số (pixel > ngưỡng)
    _, thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        raise ValueError("Không tìm thấy chữ số trong ảnh (có thể quá mờ hoặc trống)")
    x, y, w, h = cv2.boundingRect(coords)

    # Cắt vùng chứa chữ số
    digit = img[y:y+h, x:x+w]

    # Tính toán padding để căn giữa vào khung vuông
    size = max(w, h)
    delta_w = size - w
    delta_h = size - h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    digit = cv2.copyMakeBorder(digit, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    # Resize về 28x28 (dùng nội suy area để giảm hoặc cubic để tăng)
    digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)

    # Chuẩn hóa về [0,1] rồi áp dụng mean/std của MNIST
    digit = digit.astype(np.float32) / 255.0
    digit = (digit - mean) / std

    # Trả về tensor [1, 1, 28, 28]
    tensor = torch.from_numpy(digit).unsqueeze(0).unsqueeze(0)
    return tensor.to(device)

def predict_image(image_path):
    mean = 0.1307
    std = 0.3081
    tensor = preprocess_image(image_path, invert=True)
    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(dim=1).item()
        probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
    print(f"Predicted digit: {pred}")
    print(f"Confidence: {probs[pred]:.2%}")

    # (Tuỳ chọn) Hiển thị ảnh sau khi xử lý để debug
    viz = tensor.squeeze().cpu().numpy()
    viz = (viz * std + mean) * 255  # đảo ngược chuẩn hóa để hiển thị
    viz = np.clip(viz, 0, 255).astype(np.uint8)
    
# Thử với ảnh bạn có
predict_image("/Users/nguyenvokhang/Downloads/MNIST/mnist_6-300x295.png")