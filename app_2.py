import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,3,padding=1)
        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.pool = nn.MaxPool2d(2,2)

        self.fc1 =nn.Linear(16384, 128)
        self.fc2 = nn.Linear(128,10)
    
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x,1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model =CustomModel()

model.load_state_dict(torch.load("hand_model_1.pt",map_location="cpu"))
model.eval()

print("weights updated successfully")

labels ={0:"palm", 1: "I",2: "fist", 3: "fist_moved",4: "thumb", 5:"index", 6: "ok", 7:"palm_moved", 8: "c",9: "down"}

cap =cv.VideoCapture(0)


while True:
    ret, frame = cap.read()
    if not ret: break

    img = cv.resize(frame, (64,64))
    gray= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    

    gray_3ch = np.stack((gray, gray, gray), axis=-1)

    img_tensor = torch.from_numpy(gray_3ch).permute(2,0,1).float().div(255.0).unsqueeze(0)

    with torch.inference_mode():
        logits = model(img_tensor)
        _, predicted = torch.max(logits,1)

    label = labels.get(predicted.item(),"Unknown")
    display_text = f"{label}"
    

    cv.putText(frame, display_text,(50,50),cv.FONT_HERSHEY_COMPLEX,1.6,(255,0,0),2)
    cv.imshow("hand gesture ai detection", frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()