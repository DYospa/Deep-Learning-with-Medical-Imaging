import shutil
from collections import Counter
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import DenseNet121_Weights, densenet121

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RAW_MURA_TRAIN_DIR = Path("MURA-v1.1/train")
SPLIT_OUTPUT_DIR = Path("data")
CREATE_LOCAL_SPLIT = False
TEST_SIZE = 0.20
VAL_SIZE = 0.10
RANDOM_STATE = 42

BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 1e-5
IMAGE_SIZE = (160, 160)


def get_label_from_study_path(study_path: Path) -> str:
    lower_path = str(study_path).lower()
    if "positive" in lower_path:
        return "abnormal"
    if "negative" in lower_path:
        return "normal"
    raise ValueError(f"Could not infer label from path: {study_path}")


def copy_study_images(study_path: Path, split_name: str, output_root: Path) -> None:
    label = get_label_from_study_path(study_path)
    target_dir = output_root / split_name / label
    target_dir.mkdir(parents=True, exist_ok=True)

    study_prefix = study_path.parent.name + "_" + study_path.name

    for img_file in study_path.glob("*.png"):
        if img_file.name.startswith("._") or img_file.name.startswith("."):
            continue
        target_name = f"{study_prefix}_{img_file.name}"
        shutil.copy2(img_file, target_dir / target_name)


def create_mura_train_val_test_split(
    source_dir: Path,
    output_dir: Path,
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 42
) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(
            f"Source MURA directory not found: {source_dir}. "
            "Download and unzip the Stanford MURA dataset first."
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)

    study_paths = []
    study_labels = []

    for study_path in source_dir.rglob("study*"):
        if study_path.is_dir():
            try:
                label = get_label_from_study_path(study_path)
                study_paths.append(study_path)
                study_labels.append(label)
            except ValueError:
                continue

    if not study_paths:
        raise ValueError("No study folders were found in the provided MURA directory.")

    train_val_studies, test_studies, train_val_labels, _ = train_test_split(
        study_paths,
        study_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=study_labels
    )

    relative_val_size = val_size / (1.0 - test_size)

    train_studies, val_studies = train_test_split(
        train_val_studies,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=train_val_labels
    )

    for split_name, studies in [
        ("train", train_studies),
        ("val", val_studies),
        ("test", test_studies)
    ]:
        for study_path in studies:
            copy_study_images(study_path, split_name, output_dir)

    print(f"Created local split in: {output_dir}")
    print(f"Train studies: {len(train_studies)}")
    print(f"Val studies:   {len(val_studies)}")
    print(f"Test studies:  {len(test_studies)}")


train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def is_valid_image(path: str) -> bool:
    name = Path(path).name
    return not name.startswith("._") and not name.startswith(".")


def build_dataloaders(data_root: Path, batch_size: int = 32):
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"

    if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Expected train/val/test folders under {data_root}. "
            "Either create them manually or set CREATE_LOCAL_SPLIT = True."
        )

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=train_transform,
        is_valid_file=is_valid_image
    )

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=eval_transform,
        is_valid_file=is_valid_image
    )

    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=eval_transform,
        is_valid_file=is_valid_image
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": 4,
        "pin_memory": True
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def print_dataset_stats(name, dataset):
    print(f"{name} size: {len(dataset)}")
    counts = Counter(dataset.targets)
    for idx, class_name in enumerate(dataset.classes):
        print(f"  {class_name}: {counts.get(idx, 0)}")


class MedicalDenseNet121(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        base = densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.features = base.features

        for param in self.features.parameters():
            param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(1024, num_classes)

        self.gradients = None
        self.feature_maps = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features(x)
        x = torch.relu(x)
        self.feature_maps = x

        if x.requires_grad:
            x.register_hook(self.save_gradient)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)


def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion=None):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            if criterion is not None:
                total_loss += criterion(outputs, labels).item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader) if criterion is not None else None
    acc = correct / total
    return avg_loss, acc

def evaluate_with_metrics(model, loader, class_names):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

class GradCAM:
    def __init__(self, model):
        self.model = model

    def generate(self, image, class_idx=None):
        self.model.eval()

        # reset saved grads/features
        self.model.gradients = None
        self.model.feature_maps = None

        # ensure gradients enabled
        with torch.enable_grad():
            image = image.clone().detach().requires_grad_(True)

            output = self.model(image)

            if class_idx is None:
                class_idx = output.argmax(dim=1).item()

            self.model.zero_grad()
            output[0, class_idx].backward(retain_graph=True)

        if self.model.gradients is None:
            raise RuntimeError("Gradients were not captured. Hook did not fire.")

        gradients = self.model.gradients[0].detach().cpu().numpy()
        activations = self.model.feature_maps[0].detach().cpu().numpy()

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, IMAGE_SIZE)

        if cam.max() > 0:
            cam /= cam.max()

        return cam


def show_gradcam(image_tensor, cam):
    image = image_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = heatmap.astype(np.float32) / 255.0

    overlay = heatmap + image
    overlay = overlay / (overlay.max() + 1e-8)

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay[:, :, ::-1])
    plt.axis("off")
    plt.title("Grad-CAM")
    plt.show()


def main():
    if CREATE_LOCAL_SPLIT:
        create_mura_train_val_test_split(
            source_dir=RAW_MURA_TRAIN_DIR,
            output_dir=SPLIT_OUTPUT_DIR,
            test_size=TEST_SIZE,
            val_size=VAL_SIZE,
            random_state=RANDOM_STATE
        )

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = build_dataloaders(
        SPLIT_OUTPUT_DIR,
        batch_size=BATCH_SIZE
    )

    print(f"Classes: {train_dataset.classes}")
    print_dataset_stats("Train", train_dataset)
    print_dataset_stats("Val", val_dataset)
    print_dataset_stats("Test", test_dataset)

    model = MedicalDenseNet121(num_classes=len(train_dataset.classes)).to(DEVICE)

    train_counts = Counter(train_dataset.targets)
    class_weights = torch.tensor([1.4, 1.0], dtype=torch.float32, device=DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        if epoch == 5:
            print("Unfreezing DenseNet backbone...")

            for param in model.features.parameters():
                param.requires_grad = True

            optimizer = optim.Adam(model.parameters(), lr=1e-5)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        _, test_acc = evaluate(model, test_loader, criterion)

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_densenet121_model.pth")
            print("New best model saved!")

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"Test Acc:   {test_acc:.4f}")

    # ===== Training finished =====

    model.load_state_dict(torch.load("best_densenet121_model.pth", map_location=DEVICE))
    model.eval()

    # Final metrics
    evaluate_with_metrics(model, test_loader, test_dataset.classes)

    # Grad-CAM
    gradcam = GradCAM(model)

    sample_image, sample_label = test_dataset[-1]
    sample_image = sample_image.unsqueeze(0).to(DEVICE)

    cam = gradcam.generate(sample_image)

    print(f"Grad-CAM generated for label index: {sample_label}")

    show_gradcam(sample_image, cam)


if __name__ == "__main__":
    main()
