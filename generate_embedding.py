import os
import cv2
import clip
import torch
import pandas as pd
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from tqdm import tqdm

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

preprocess_transforms = Compose([
        Resize((300, 300), Image.BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize(
            (0.5,),  # Single channel mean for grayscale image
            (0.5,)   # Single channel std for grayscale image
        )
    ])

def get_clip_embeddings(file_path='test.png'):
    image = Image.open(file_path)
    image_input = preprocess_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embeddings = model.encode_image(image_input)
    return list(embeddings.squeeze().cpu().numpy())

def get_clip_text_embeddings(prompt):
    prompt_tensor = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        embeddings = model.encode_text(prompt_tensor)
    return list(embeddings.squeeze().cpu().numpy())
    
def get_clip_embeddings_dir(image_dir):
    image_embeddings = []
    # Preprocess the image
    
    for file in tqdm(os.listdir(image_dir), desc="Processing Images"):
        file_path = os.path.join(image_dir, file)

        if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # Load the image
            image = Image.open(file_path)

            # Preprocess the image
            image_input = preprocess_transforms(image).unsqueeze(0).to(device)

            # Get CLIP embeddings
            with torch.no_grad():
                embeddings = model.encode_image(image_input)

            image_name = os.path.splitext(file)[0]
            image_embeddings.append([image_name] + list(embeddings.squeeze().cpu().numpy()))

    return image_embeddings

# # Replace with the path to your dataset directory
# image_dir = 'Images/train'

# # Get the embeddings
# image_embeddings = get_clip_embeddings_dir(image_dir)


# columns=['Image Name']
# columns.extend([f'embedding_dim_{i}'for i in range(512)])
# # Save the embeddings as a CSV file
# embeddings_df = pd.DataFrame(image_embeddings, columns=columns)
# embeddings_df.to_csv('./Datasets/train_embeddings.csv', index=False)
