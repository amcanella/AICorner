from transformers import AutoFeatureExtractor, ResNetForImageClassification   
import torch 
from datasets import load_dataset

from IPython.display import Image
from PIL import Image as PILImage

'''dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

imageDOS= PILImage.open('C:/Repos/aiCorner/finger.jpg')'''

feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-152")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-152")

def predictor(image):

    inputs = feature_extractor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    # model predicts one of the 1000 ImageNet classes
    predicted_label = logits.argmax(-1).item()
    
    return predicted_label


#print(f'el resultado es: {model.config.id2label[predictor(imageDOS)]}')