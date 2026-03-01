import cv2 as cv
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
class CustomModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = self.conv(3,32)
        self.conv2 = self.conv(32,64)
        self.conv3 = self.conv(64,128)

        self.up1 = nn.ConvTranspose2d(128,64,2,stride=2)
        self.up2 = nn.ConvTranspose2d(64,32,2,stride=2)
        self.up3 = nn.ConvTranspose2d(32,1,2,stride=2)
    def conv(self,inp_c, out_c):
        return nn.Sequential(
            nn.Conv2d(inp_c,out_c,3,padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

    def forward(self, x):

        cnn_out = self.conv3(self.conv2(self.conv1(x)))

        y = self.up3(self.up2(self.up1(cnn_out)))

        return y
    
model =CustomModel()
model.load_state_dict(torch.load("lane_detection_hehe_1.pt",map_location="cpu"))
model.eval()
cap = cv.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret: break

    img = cv.resize(frame,(128,128))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    gray_3ch = np.stack((gray,gray,gray),axis=-1)

    img_tensor =  torch.from_numpy(gray_3ch).permute(2,0,1).float().div(255.0)

    img_tensor = (img_tensor - 0.5)/ 0.5

    img_tensor = img_tensor.unsqueeze(0)

    with torch.inference_mode():

        output = model(img_tensor)
        lane_mask = torch.sigmoid(output)
        lane_mask = lane_mask.squeeze().numpy()
        lane_mask = (lane_mask * 255).astype(np.uint8)
        lane_mask_color = np.zeros_like(frame)
        lane_resized = cv.resize(lane_mask, (frame.shape[1], frame.shape[0]))
        lane_mask_color[:, :, 2] = lane_resized
        
    overlay = cv.addWeighted(frame, 0.8, lane_mask_color, 1.05, 1)

    cv.imshow("Lane Detection", overlay)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
