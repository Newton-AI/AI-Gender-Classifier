import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

import os
import warnings
from PIL import Image
from cvzone.FaceDetectionModule import FaceDetector

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18

warnings.filterwarnings('ignore')

classes = ['Female', 'Male']
def build_classifier():
  classifier = resnet18()
  classifier.fc = nn.Linear(classifier.fc.in_features, len(classes))
  return classifier

def crop_face(img, boxes, margin=0.2):
    faces = []
    for box in boxes:
        x1, y1, w, h = box['bbox']
        x2, y2 = x1 + w, y1 + h
        x1_mg, x2_mg = int(x1 * margin), int(x2 * margin)
        y1_mg, y2_mg = int(y1 * margin), int(y2 * margin)
        face = img[y1-y1_mg:y2+y2_mg, x1-x1_mg:x2+x2_mg]
        faces.append(face)
    return faces

folder_path = os.path.dirname(__file__)
detector = FaceDetector()
classifier = build_classifier()
path = os.path.join(folder_path, 'resnet-18.pth')
classifier.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
classifier.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.set_page_config('Face Gender Classifier')
st.title('Face Gender Classifier')
img_buffer = st.camera_input('Classifying Gender..')

if img_buffer:
    img = np.array(Image.open(img_buffer))
    boxes = detector.findFaces(img, draw=False)[1]

    if boxes:
        faces = crop_face(img, boxes)
        fig, ax = plt.subplots(len(faces))

        for i in range(len(faces)):
            PIL_img = Image.fromarray(faces[i])
            transformed_img = transform(PIL_img).unsqueeze(0)

            with torch.no_grad():
                output = classifier(transformed_img)
                _, predictions = torch.max(output, 1)
                label = classes[predictions.item()]
            
            if len(faces) == 1:
                ax.imshow(PIL_img)
                ax.set_title(label)
                ax.axis("off")
            else:
                ax[i].imshow(PIL_img)
                ax[i].set_title(label)
                ax[i].axis("off")
        
        st.pyplot(fig)

    else:
        st.error('No Face Detected!')
