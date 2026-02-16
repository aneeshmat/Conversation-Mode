import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model import VoiceClassifier

# Load your extracted features and labels
X = torch.load("features.pt")   # Nx18
y = torch.load("labels.pt")     # Nx1

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = VoiceClassifier(input_dim=X.shape[1])
opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()

for epoch in range(20):
    for xb, yb in loader:
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
    print("epoch", epoch, "loss", float(loss))

torch.save(model.state_dict(), "voice_classifier.pt")
print("Model saved.")

