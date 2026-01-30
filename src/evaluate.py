import os
import json
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


DATA_DIR = "data"
MODEL_PATH = os.environ.get("MODEL_PATH", "model/image_classifier.pth")
RESULTS_DIR = "results"
BATCH_SIZE = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


val_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "val"),
    transform=val_transforms
)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

num_classes = len(val_dataset.classes)


model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()


all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
conf_matrix = confusion_matrix(all_labels, all_preds).tolist()

metrics = {
    "accuracy": accuracy,
    "precision_weighted": precision,
    "recall_weighted": recall,
    "confusion_matrix": conf_matrix
}


os.makedirs(RESULTS_DIR, exist_ok=True)
metrics_path = os.path.join(RESULTS_DIR, "metrics.json")

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

print("Evaluation completed.")
print(f"Metrics saved to {metrics_path}")
