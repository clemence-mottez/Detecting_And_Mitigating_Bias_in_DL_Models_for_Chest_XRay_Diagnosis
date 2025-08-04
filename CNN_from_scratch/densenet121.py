# Train our own DenseNet121 model on CheXpert data

import os
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from torch.utils.data import WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau


# ------------------ Configuration ------------------
CHEXPERT_IMAGE_ROOT = "../../../lfay/CheXpert-v1.0-512/images"
CSV_TRAIN_PATH      = "../data/processed_chexpert_data/chexpert_plus_train.csv"
CSV_VALID_PATH      = "../data/processed_chexpert_data/chexpert_plus_valid.csv"
CSV_TEST_PATH       = "../data/processed_chexpert_data/chexpert_plus_test.csv"
CHEXNET_WEIGHTS_PATH = "weights.h5"  # Keras-format CheXNet weights

NUM_CLASSES   = 4
BATCH_SIZE    = 16
NUM_EPOCHS    = 10
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

dataloaders = {
    'train': DataLoader(
        datasets['train'],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    ),
    'val': DataLoader(
        datasets['val'],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    ),
    'test': DataLoader(
        datasets['test'],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
}


dataset_sizes = {x: len(datasets[x]) for x in datasets}



# ------------------ Weight Loading ------------------
def load_keras_h5_weights_into_pytorch(model, h5_path):
    """
    Load Keras .h5 CheXNet weights into a torchvision DenseNet-121 PyTorch model.
    This function maps layer names and copies weights/biases accordingly.
    """
    f = h5py.File(h5_path, 'r')
    state_dict = model.state_dict()

    for layer_name, param in state_dict.items():
        # Skip classifier layers
        if layer_name.startswith('classifier'):
            continue
        # Derive Keras group/key names
        keras_name = layer_name
        # BatchNorm mappings
        if 'running_mean' in layer_name:
            group = layer_name.replace('features.', '').replace('running_mean', 'moving_mean')
            weight_key = f"{group}:0"
        elif 'running_var' in layer_name:
            group = layer_name.replace('features.', '').replace('running_var', 'moving_variance')
            weight_key = f"{group}:0"
        elif 'bn' in layer_name and 'weight' in layer_name:
            group = layer_name.replace('features.', '').replace('.weight', '')
            weight_key = f"{group}/gamma:0"
        elif 'bn' in layer_name and 'bias' in layer_name:
            group = layer_name.replace('features.', '').replace('.bias', '')
            weight_key = f"{group}/beta:0"
        elif 'bias' in layer_name:
            group = layer_name.replace('features.', '').replace('.bias', '')
            weight_key = f"{group}/bias:0"
        elif 'weight' in layer_name:
            group = layer_name.replace('features.', '').replace('.weight', '')
            weight_key = f"{group}/kernel:0"
        else:
            continue

        try:
            np_w = f[weight_key][:]
            # For conv weights: Keras uses (H, W, in, out)
            if np_w.ndim == 4:
                np_w = np_w.transpose(3, 2, 0, 1)
            # For linear weights: (in, out)
            if np_w.ndim == 2 and param.ndim == 2:
                np_w = np_w.T
            # Copy into param
            param.copy_(torch.from_numpy(np_w))
        except KeyError:
            # Missing in .h5, skip
            pass

    f.close()
    return model

# ------------------ Model Setup ------------------
model = models.densenet121(pretrained=False)
num_ftrs = model.classifier.in_features
# model.classifier = nn.Linear(num_ftrs, NUM_CLASSES)
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, NUM_CLASSES)
)
model = model.to(DEVICE)

# Load Keras-format CheXNet weights
if CHEXNET_WEIGHTS_PATH.endswith('.h5'):
    model = load_keras_h5_weights_into_pytorch(model, CHEXNET_WEIGHTS_PATH)
else:
    # Fall back to PyTorch .pth loading
    checkpoint = torch.load(CHEXNET_WEIGHTS_PATH, map_location=DEVICE)
    state = checkpoint.get('state_dict', checkpoint)
    filtered = {k.replace('module.', ''):v for k,v in state.items() if not k.startswith('classifier')}
    model.load_state_dict(filtered, strict=False)

# ------------------ Loss, Optimizer, Scheduler ------------------
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5, betas=(0.9, 0.999))
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
scheduler = ReduceLROnPlateau(optimizer,
                              mode='min',
                              factor=0.5,
                              patience=2,
                              verbose=True)

# ------------------ Training & Validation ------------------
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_model(model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS, patience=3):
    """
    scheduler: should be a ReduceLROnPlateau instance
    patience:  number of epochs with no val‐loss improvement before stopping
    """
    best_wts = model.state_dict()
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, running_count = 0.0, 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_count += inputs.size(0)

            epoch_loss = running_loss / running_count
            print(f'{phase} Loss: {epoch_loss:.4f}')

            # -- validation logic: track best, early‐stop counter, scheduler step
            if phase == 'val':
                # LR scheduler step on validation loss
                scheduler.step(epoch_loss)

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_wts = model.state_dict()
                    epochs_no_improve = 0
                    print("  → New best val loss!")
                else:
                    epochs_no_improve += 1
                    print(f"  → No improvement for {epochs_no_improve}/{patience} epochs.")

        print()

        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered (no improvement in {patience} epochs).")
            break

    print(f'Best val Loss: {best_loss:.4f}')
    model.load_state_dict(best_wts)
    return model


# ------------------ Main ------------------
def main():
    trained = train_model(model, criterion, optimizer, scheduler)
    # Test evaluation
    trained.eval()
    test_loss, test_count = 0.0, 0
    for inputs, labels in dataloaders['test']:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        with torch.no_grad():
            outputs = trained(inputs)
            loss = criterion(outputs, labels)
        test_loss += loss.item()*inputs.size(0)
        test_count += inputs.size(0)
    print(f'Test Loss: {test_loss/test_count:.4f}')
    # Save final model
    torch.save(trained.state_dict(), 'densenet121_chexpert_4class_10epochs.pth')

if __name__ == '__main__':
    main()
