import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

'''Le script vise à détecter les contours de brins de paille dans une image donnée. Le modèle de deep learning U2NET est utilisé pour générer un masque de segmentation de l'image. 
Ensuite, le script affine ce masque pour identifier les contours des brins de paille et calcule des statistiques (longueurs des brins).
 En plus, une visualisation Grad-CAM est proposée pour comprendre quelles parties de l'image influencent le plus la prédiction du modèle.'''


# Définition de la classe REBNCONV
class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))

def _upsample_like(src, tar):
    return F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=False)

# Définition de la classe RSU5
class RSU5(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin

# Définition de la classe U2NET
class U2NET(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET, self).__init__()

        self.stage1 = RSU5(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU5(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU5(256, 128, 512)

        self.stage5 = RSU5(512, 256, 512)

        # decoder
        self.stage4d = RSU5(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU5(256, 32, 64)
        self.stage1d = RSU5(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(5*out_ch, out_ch, 1)

    def forward(self, x):
        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)

        #stage 5
        hx5 = self.stage5(hx4)

        #decoder
        hx4d = self.stage4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        #side output
        d1 = self.side1(hx1d)
        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)
        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)
        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)
        d5 = self.side5(hx5)
        d5 = _upsample_like(d5, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5), 1))

        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5)

# Paramètres
RESIZE_DIM = 256
STRAW_THRESHOLD = 0.65
MIN_CONTOUR_AREA = 30
CONTOUR_THICKNESS = 1
KERNEL_SIZE = 3

#Chargement du modèle
@st.cache_resource
def load_model():
    model = U2NET(in_ch=3, out_ch=1)
    model.load_state_dict(torch.load('u2net_best.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

#Prétraitement de l'image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((RESIZE_DIM, RESIZE_DIM)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

#Prédiction du masque
def predict_mask(model, image):
    with torch.no_grad():
        output = model(image)
    return torch.sigmoid(output[0]).squeeze().cpu().numpy()

def refine_straw_mask(pred_mask):
    binary_mask = (pred_mask > STRAW_THRESHOLD).astype(np.uint8)
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    eroded = cv2.erode(binary_mask, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    dilated = cv2.bitwise_and(dilated, binary_mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < MIN_CONTOUR_AREA:
            dilated[labels == i] = 0
    return dilated * 255

def get_straw_contours(straw_mask):
    contours, _ = cv2.findContours(straw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    return contours

def calculate_straw_lengths(contours):
    return [cv2.arcLength(cnt, True) for cnt in contours]

#Visualisation Grad-CAM
def get_gradcam(model, input_tensor, target_layer):
    model.eval()
    model.zero_grad()
    
    activations = {}
    gradients = {}
    
    def save_activation(name):
        def hook(module, input, output):
            activations[name] = output
        return hook
    
    def save_gradient(name):
        def hook(module, grad_input, grad_output):
            gradients[name] = grad_output[0]
        return hook
    
    handle_activ = target_layer.register_forward_hook(save_activation('target'))
    handle_grad = target_layer.register_backward_hook(save_gradient('target'))
    
    output = model(input_tensor)
    
    model.zero_grad()
    output[0].sum().backward()
    
    gradients = gradients['target']
    activations = activations['target']
    
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    gradcam = torch.sum(weights * activations, dim=1, keepdim=True)
    gradcam = F.relu(gradcam)
    gradcam = F.interpolate(gradcam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
    gradcam = gradcam.squeeze().cpu().detach().numpy()
    
    gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min())
    
    handle_activ.remove()
    handle_grad.remove()
    
    return gradcam

def apply_gradcam_visualization(image, gradcam, contrast, colormap):
    plt.figure(figsize=(10, 5))
    plt.imshow(image)
    gradcam_adjusted = np.power(gradcam, contrast)
    gradcam_adjusted = (gradcam_adjusted - gradcam_adjusted.min()) / (gradcam_adjusted.max() - gradcam_adjusted.min())
    plt.imshow(gradcam_adjusted, cmap=colormap, alpha=0.7)
    plt.title("Grad-CAM")
    plt.axis('off')
    cbar = plt.colorbar()
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Faible', 'Moyen', 'Élevé'])
    cbar.set_label('Importance pour la prédiction', rotation=270, labelpad=15)
    return plt


def main():
    st.title("Détection de brins de paille avec U2NET")

    model = load_model()

    uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Image originale", use_column_width=True)

        processed_image = preprocess_image(image)
        pred_mask = predict_mask(model, processed_image)
        refined_mask = refine_straw_mask(pred_mask)
        contours = get_straw_contours(refined_mask)

        # Affichage du masque prédit
        fig, ax = plt.subplots()
        ax.imshow(pred_mask, cmap='gray')
        ax.set_title("Masque prédit")
        ax.axis('off')
        st.pyplot(fig)

        # Affichage des contours
        image_with_contours = np.array(image.resize((RESIZE_DIM, RESIZE_DIM)))
        cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), CONTOUR_THICKNESS)
        st.image(image_with_contours, caption="Contours des brins de paille", use_column_width=True)

        # Distribution des longueurs
        lengths = calculate_straw_lengths(contours)
        fig, ax = plt.subplots()
        ax.hist(lengths, bins=30, edgecolor='black')
        ax.set_xlabel("Longueur des brins de paille (pixels)")
        ax.set_ylabel("Fréquence")
        ax.set_title("Distribution des longueurs des brins de paille")
        st.pyplot(fig)

        # Grad-CAM avec contrôles dynamiques
        target_layer = model.stage1.rebnconv1.conv_s1
        gradcam = get_gradcam(model, processed_image, target_layer)
        
        st.subheader("Visualisation Grad-CAM")
        contrast = st.slider("Contraste", min_value=0.1, max_value=5.0, value=3.0, step=0.1)
        colormap = st.selectbox("Palette de couleurs", options=['inferno', 'viridis', 'plasma', 'magma', 'hot', 'cool'])
        
        gradcam_fig = apply_gradcam_visualization(np.array(image.resize((RESIZE_DIM, RESIZE_DIM))), gradcam, contrast, colormap)
        st.pyplot(gradcam_fig)

if __name__ == "__main__":
    main()