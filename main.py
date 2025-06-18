import streamlit as st
import cv2
import numpy as np
import torch
from Resnet34Unet import ResnetSuperVision  # Import your model class
from torchvision.transforms.functional import to_tensor
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import base64

plt.rcParams['axes.facecolor'] = 'black'
@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()
def header(url):
     st.markdown(f'<h1 style="background-color:#070E21;color:#FFFFFF;font-size:40px;border-radius:2%;">{url}</h1>', unsafe_allow_html=True)

img = get_img_as_base64("background.png")
# <a href="https://ibb.co/G3rQ6R9"><img src="https://i.ibb.co/jZjkNDz/Adobe-Stock-492863232-Preview.jpg" alt="Adobe-Stock-492863232-Preview" border="0"></a>

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://i.ibb.co/mHy77nC/bbhhhbhh.jpg");
background-size: 100%;
background-position: 60% 50%;
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""
header("IDENTIFICATION OF LUNG DETERIORATION BY IRREVERSIBLE PULMONARY FIBROSIS USING DEEP NEURAL NETWORKS")
st.markdown(page_bg_img, unsafe_allow_html=True)


# Load the trained model
model = ResnetSuperVision(2, backbone_arch='resnet34')  # Assuming 2 classes (lung and fibrosis)
model.load_state_dict(torch.load("resnet34_fib_best.pth",map_location=torch.device('cpu')))
# model.eval()

# Function to perform inference
@st.cache_resource
def predict(image_np):
    # Convert the image to RGB if it's grayscale
    if image_np.ndim == 2:  # Grayscale image
        image_np = np.stack([image_np] * 3, axis=-1)  # Convert to RGB

    # Normalize the image and convert to tensor
    normalized_image = image_np.astype(np.float32) / 255.0
    input_tensor = to_tensor(normalized_image).unsqueeze(0)

    # Perform prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor, get_fet=True)  # Note the comma after 'output'

        lung_mask = output[0][0, 1] > 0.9
        fibrosis_mask = output[0][0, 0] > 0.9
        lung_mask_np = lung_mask.cpu().numpy()
        fibrosis_mask_np = fibrosis_mask.cpu().numpy()
        # Calculate percentage of fibrosis
        lung_area = np.sum(lung_mask_np)
        fibrosis_area = np.sum(np.logical_and(fibrosis_mask_np, lung_mask_np))
        
        if lung_area > 0:
            fibrosis_percentage = (fibrosis_area / lung_area) * 100
        else:
            fibrosis_percentage = 0.0

    return lung_mask, fibrosis_mask,fibrosis_percentage

# Streamlit App
def main():
    # st.title("Lung and Fibrosis Segmentation App")
    

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image file
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # Display the original image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Process the image and get segmentation masks
        lung_mask, fibrosis_mask ,fibrosis_percentage= predict(image_np)

        # Display the segmented masks
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        # fig.patch.set_facecolor('black')
        ax[0].imshow(image_np)
        ax[0].imshow(lung_mask, cmap='hot', alpha=0.5)
        ax[0].set_title("Lung Segmentation")
        ax[0].axis('off')

        ax[1].imshow(image_np)
        ax[1].imshow(fibrosis_mask, cmap='hot', alpha=0.7)
        # ax[1].set_title(f"Fibrosis Segmentation ({fibrosis_percentage:.2f}% of Lung Area)")
        ax[1].set_title("Fibrosis Segmentation")
        ax[1].axis('off')

        st.pyplot(fig)
        if fibrosis_percentage==0:
            st.success("No Pulmonary Fibrosis Detected")
        else:
            st.info(f"Fibrosis Segmentation ({ fibrosis_percentage:.2f}% of Lung Area )")

if __name__ == "__main__":
    main()
