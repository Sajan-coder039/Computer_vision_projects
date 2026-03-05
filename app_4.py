import cv2 as cv
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),                                     
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = self.convv(3,64)
        self.conv2 =  self.convv(64,32)
        self.conv3 = self.convv(32,16)
        self.fc1 = nn.Linear(12544,1024)
        self.fc2 = nn.Linear(1024,5)
        self.relu = nn.ReLU()

    def convv(self,input,output):
        return nn.Sequential(
            nn.Conv2d(input, output, kernel_size=3, stride=2, padding= 1),
            nn.BatchNorm2d(output),
            nn.Dropout2d(0.2),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        x = x.view(x.size(0),-1)
        x = self.fc2(self.relu(self.fc1(x)))
        return x

model = CustomModel()
model.load_state_dict(torch.load("kartik_1.pt",map_location = torch.device("cpu")) )
model.eval()
cap =  cv.VideoCapture(0)

label = {0:"daisy",1: "dandelion",2:"rose",3:"sunflower",4:"tulip"}

while True:

    ret, frame = cap.read()

    if not ret: break

    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    img_tensor = transform(img).unsqueeze(0)

    with torch.inference_mode():
        output = model(img_tensor)

    conf = torch.softmax(output, dim=1).max().item()

    label_= output.argmax(1).item()

    display = label[label_]

    cv.putText(frame,f"flower:: {display} ({conf:.0%})", (50,50),cv.FONT_HERSHEY_COMPLEX,1.2,(255,0,0),1)
    cv.imshow("Flower Detection", frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()




