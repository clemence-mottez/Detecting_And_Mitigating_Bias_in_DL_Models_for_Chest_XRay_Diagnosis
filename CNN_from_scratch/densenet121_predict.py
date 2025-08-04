# Extract predictions from images using our finetuned DenseNet121 model

import os
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# ------------------ Configuration ------------------
CHEXPERT_IMAGE_ROOT = "../../../lfay/CheXpert-v1.0-512/images"
CSV_TRAIN_PATH      = "../data/processed_chexpert_data/chexpert_plus_240401_train.csv"
CSV_VALID_PATH      = "../data/processed_chexpert_data/chexpert_plus_240401_valid.csv"
CSV_TEST_PATH       = "../data/processed_chexpert_data/chexpert_plus_240401_valid.csv"
# CHEXNET_WEIGHTS_PATH = "weights.h5"  # Keras-format CheXNet weights
MY_WEIGHTS_PATH     = "densenet121_chexpert_4class_10epochs.pth"  

NUM_CLASSES   = 4
BATCH_SIZE    = 16
NUM_EPOCHS    = 20
LEARNING_RATE = 1e-4
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------ Dataset ------------------
class CheXpertDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.labels = ['Cardiomegaly', 'Lung Opacity', 'Edema', 'Pleural Effusion']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['path_to_image'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        labels = torch.FloatTensor(row[self.labels].values.astype(float))
        return image, labels

# ------------------ Transforms ------------------
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),  
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229])
    ])
}

# ------------------ DataLoaders ------------------
datasets = {x: CheXpertDataset(
                csv_file=path,
                img_dir=CHEXPERT_IMAGE_ROOT,
                transform=data_transforms[x])
            for x, path in [('train', CSV_TRAIN_PATH), ('val', CSV_VALID_PATH), ('test', CSV_TEST_PATH)]}

dataloaders = {x: DataLoader(datasets[x], batch_size=BATCH_SIZE, shuffle=(x=='train'), num_workers=4)
               for x in datasets}

dataset_sizes = {x: len(datasets[x]) for x in datasets}


# ------------------ Model Setup ------------------
model = models.densenet121(pretrained=False)
num_ftrs = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, NUM_CLASSES)
)
model = model.to(DEVICE)


# ------------------ Inference ------------------
def predict_on_images(model, csv_path, img_dir, model_path, transform, device, output_csv='predictions_valid10.csv'):
    # Load saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    df = pd.read_csv(csv_path)
    preds = []
    for _, row in df.iterrows():
        img_path = os.path.join(img_dir, row['path_to_image'])
        img = Image.open(img_path).convert('RGB')
        inp = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp)
            prob = torch.sigmoid(out).cpu().numpy().flatten().tolist()
        preds.append([row['path_to_image']] + prob)

    out_df = pd.DataFrame(preds, columns=['path_to_image'] + datasets['train'].labels)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")

# ------------------ Main ------------------
def main():
    predict_on_images(
        model,
        CSV_TEST_PATH,
        CHEXPERT_IMAGE_ROOT,
        MY_WEIGHTS_PATH,
        data_transforms['test'],
        DEVICE
    )

if __name__ == '__main__':
    main()
