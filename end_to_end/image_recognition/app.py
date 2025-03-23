import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn

# Cache the model so it is loaded only once.
@st.cache_resource
def load_model():
    # Initialize a ResNet18 model and update the final layer to match your dataset.
    num_classes = 10  # Change if you have a different number of classes.
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Load your trained weights; adjust the path as necessary.
    model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Define the transform to match what was used during training.
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet18 expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Define the class names (example for CIFAR10).
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

st.title("ResNet Image Classifier")
st.write("Upload an image and the model will predict its label along with a confidence score.")

# File uploader widget.
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Open and display the image.
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess the image.
    input_image = transform(image).unsqueeze(0)  # add batch dimension
    
    with torch.no_grad():
        output = model(input_image)
        # Compute softmax to get class probabilities.
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, dim=0)
        # Convert the predicted label to CamelCase.
        predicted_label = class_names[predicted_idx].title()
    
    st.write(f"**Predicted Label:** {predicted_label}")
    st.write(f"**Confidence:** {confidence.item() * 100:.2f}%")