import cv2 as cv
from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Load the architecture (the "body")
# Use 'yolo11n.pt' or the specific version you trained with
model = YOLO("yolo11n.pt") 
class CustomModel(nn.Module):
    def __init__(self,):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,3,padding=1)
        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.pool= nn.MaxPool2d(2,2)

        self.fc1= nn.Linear(200704,128)
        self.fc2 =nn.Linear(128,2)

    def forward(self, x):
        x =self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x =self.fc2(x)
        return x
# 2. Load your custom weights (the "brain")
# This "injects" your trained math into the YOLO structure
# Change this:
# weights = torch.load("model_1.pt")

# To this:
model = CustomModel()
model.load_state_dict(torch.load("model_1.pt", map_location='cpu'))
model.eval()

print("Weights loaded successfully!")

cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. Pre-process the frame
    # Make sure 'img_size' matches what you used in training (e.g., 224)
    img_size = 224 
    resized = cv.resize(frame, (img_size, img_size))
    
    # Convert BGR (OpenCV) to RGB
    rgb = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
    
    # Transform to Tensor: Shape (H, W, C) -> (C, H, W) and scale to 0-1
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    
    # Add Batch Dimension: (C, H, W) -> (1, C, H, W)
    input_tensor = tensor.unsqueeze(0)

    # 2. Run Inference
    with torch.no_grad(): # No need to track gradients for detection
        output = model(input_tensor)
        
    # 3. Handle the Output
    # Since this is a custom CNN, 'output' is likely a tensor of probabilities
    # We find the class with the highest probability
    _, predicted = torch.max(output, 1)
    class_id = predicted.item()
    
    # Define your labels based on your training
    labels = {0: "Mask", 1: "No Mask"}
    label = labels.get(class_id, "Unknown")

    # 4. Draw the result
    color = (0, 255, 0) if class_id == 0 else (0, 0, 255)
    cv.putText(frame, label, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv.imshow("Custom Mask Detector", frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv.destroyAllWindows()