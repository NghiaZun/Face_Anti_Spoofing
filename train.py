import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from casia_dataset import CasiaDataset, BalancedSampler
from resnet15 import ResNet15
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
csv_file = 'face-check/data/labels.csv'
root_dir = 'face-check/data/casia_mfsd'

df = pd.read_csv(csv_file)
labels = [1 if l != 'real' else 0 for l in df['label']]
dataset = CasiaDataset(csv_file, root_dir)
sampler = BalancedSampler(labels, batch_size=64)
loader = DataLoader(dataset, batch_size=64, sampler=sampler)

model = ResNet15().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

for epoch in range(30):
    model.train()
    total, correct = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = outputs.argmax(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    acc = 100 * correct / total
    print(f"Epoch {epoch+1}: Loss {loss.item():.4f}, Acc {acc:.2f}%")
    scheduler.step()

torch.save(model.state_dict(), "resnet15_casia.pth")
