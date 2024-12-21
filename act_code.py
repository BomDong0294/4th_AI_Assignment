import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from random import choices
from torchvision.models import VGG16_Weights, ResNet50_Weights, GoogLeNet_Weights, AlexNet_Weights, DenseNet161_Weights
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter




# 데이터셋 클래스 정의
class RockPaperScissorsDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_validation=False):
        self.data = []
        self.labels = []
        self.transform = transform
        self.label_map = {"rock": 0, "paper": 1, "scissors": 2}

        if is_validation:
            # Validation 데이터: 폴더 구분 없이 단순 이미지
            for img_name in os.listdir(root_dir):
                img_path = os.path.join(root_dir, img_name)
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    image = cv2.imread(img_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (224, 224))
                    self.data.append(image)
                    for label in self.label_map.keys():
                        if label in img_name.lower():
                            self.labels.append(self.label_map[label])
                            break
        else:
            # Train/Test 데이터: 폴더별 클래스 구분
            for label in self.label_map.keys():
                label_dir = os.path.join(root_dir, label)
                for img_name in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, img_name)
                    image = cv2.imread(img_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (224, 224))
                    self.data.append(image)
                    self.labels.append(self.label_map[label])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label





# 데이터 전처리 정의
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 데이터셋 경로
data_path_train = "./rock_paper_scissors/train"
data_path_test = "./rock_paper_scissors/test"
data_path_validation = "./rock_paper_scissors/validation"

train_dataset = RockPaperScissorsDataset(data_path_train, transform=transform)
test_dataset = RockPaperScissorsDataset(data_path_test, transform=transform)
validation_dataset = RockPaperScissorsDataset(data_path_validation, transform=transform, is_validation=True)

# DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)






# 모델 선택 함수 (VGG16, ResNet50, GoogLeNet, AlexNet, DenseNet161)
def get_model(model_name, num_classes):
    if model_name == 'vgg16':
        weights = VGG16_Weights.IMAGENET1K_V1
        model = models.vgg16(weights=weights)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == 'resnet50':
        weights = ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'googlenet':
        weights = GoogLeNet_Weights.IMAGENET1K_V1
        model = models.googlenet(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'alexnet':
        weights = AlexNet_Weights.IMAGENET1K_V1
        model = models.alexnet(weights=weights)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == 'densenet161':
        weights = DenseNet161_Weights.IMAGENET1K_V1
        model = models.densenet161(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError("Unsupported model name. Choose from 'vgg16', 'resnet50', 'googlenet', 'alexnet', 'densenet161'.")
    return model






# 디바이스 설정 (model list : resnet50, alexnet, densenet161, googlenet, vgg16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model('googlenet', num_classes=3).to(device)

# 데이터 분포 확인
print("Train class distribution:", Counter(train_dataset.labels))
print("Test class distribution:", Counter(test_dataset.labels))

# 클래스 가중치 계산
class_weights = compute_class_weight('balanced', classes=np.unique(train_dataset.labels), y=train_dataset.labels)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

# 크로스 엔트로피 손실 함수에 클래스 가중치 추가
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001)





# 모델 학습 함수
def train_model(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

# 모델 평가 함수
def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds

# 모델 저장 함수
def save_model(model, file_name):
    torch.save(model.state_dict(), file_name)
    print(f"Model saved to {file_name}")

# 모델 불러오기 함수
def load_model(model_name, num_classes, file_name):
    model = get_model(model_name, num_classes)
    model.load_state_dict(torch.load(file_name))
    print(f"Model loaded from {file_name}")
    return model

# 교차 검증 (Stratified K-Fold)
def cross_validation(model_name, dataset, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(dataset.data, dataset.labels)):
        print(f"Fold {fold + 1}/{n_splits}")
        
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

        model = get_model(model_name, num_classes=3).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(3):  # 짧은 Epoch 수로 제한
            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}")

        true_labels, preds = evaluate_model(model, val_loader, device)
        f1 = f1_score(true_labels, preds, average='weighted')
        accuracy = accuracy_score(true_labels, preds)
        print(f"Fold {fold + 1} - F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}")
        fold_results.append((f1, accuracy))
    
    avg_f1 = np.mean([x[0] for x in fold_results])
    avg_acc = np.mean([x[1] for x in fold_results])
    print(f"Cross-Validation Results: Avg F1 = {avg_f1:.4f}, Avg Accuracy = {avg_acc:.4f}")

# 부트스트래핑
def bootstrap_evaluation(model, dataset, num_samples=1000):
    bootstrap_results = []
    for _ in range(num_samples):
        sample_indices = choices(range(len(dataset)), k=len(dataset))
        sample_subset = Subset(dataset, sample_indices)
        sample_loader = DataLoader(sample_subset, batch_size=32, shuffle=True)

        true_labels, preds = evaluate_model(model, sample_loader, device)
        f1 = f1_score(true_labels, preds, average='weighted')
        bootstrap_results.append(f1)

    mean_f1 = np.mean(bootstrap_results)
    ci_lower = np.percentile(bootstrap_results, 2.5)
    ci_upper = np.percentile(bootstrap_results, 97.5)
    print(f"Bootstrap Results: Mean F1 = {mean_f1:.4f}, 95% CI = ({ci_lower:.4f}, {ci_upper:.4f})")





# 실행, 초기 평가 (경고 제거)
print("Initial Performance Evaluation on Test Set:")
true_labels, preds = evaluate_model(model, test_loader, device)
print(classification_report(true_labels, preds, zero_division=0))


print("\nStarting Cross-Validation:")
cross_validation('googlenet', train_dataset)

print("\nStarting Bootstrap Evaluation:")
bootstrap_evaluation(model, validation_dataset)





# 모델 저장 (model list : resnet50, alexnet, densenet161, googlenet, vgg16)
save_model(model, "rock_paper_scissors_googlenet.pth")

