import os
import sys

sys.path.append("../../src")
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from time import sleep
from torch.utils.data import DataLoader
from pytorch_dataloader_helper import CustomAudioDataset
from singing_detect_arch import SingingDetectNet

# Model presets
BATCH_SIZE = 25
NUM_EPOCHS = 100
DEVICE = "cuda:0"
START_LEARNING_RATE = 1e-3
model_save_dir = "../../fixtures"
model_filename = "singing_detect.pth"
model_filepath = os.path.join(model_save_dir, model_filename)

# Dataloader
train_dataloader = DataLoader(
    CustomAudioDataset(training=True),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=10,
)
test_dataloader = DataLoader(
    CustomAudioDataset(training=False),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=10,
)
# Call model
def softmax_focal_loss(x, target, gamma=2.0, alpha=0.25):
    n = x.shape[0]
    device = target.device
    range_n = torch.arange(0, n, dtype=torch.int64, device=device)

    pos_num = float(x.shape[1])
    p = torch.softmax(x, dim=1)
    p = p[range_n, target]
    loss = -((1 - p) ** gamma) * alpha * torch.log(p)
    return torch.sum(loss) / pos_num


torch.cuda.empty_cache()
net = SingingDetectNet().to(DEVICE)
criterion = softmax_focal_loss
optimizer = optim.Adam(net.parameters(), lr=START_LEARNING_RATE, weight_decay=0.02)
lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.98)

# Train 100 epochs
def acc_stats(net, dataloader):
    net.eval()
    with torch.no_grad():
        correct, total = 0, 0
        fp, fn = 0, 0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data["image"].to(DEVICE), data["is_speech"].to(DEVICE)
            pred_y = torch.max(net(inputs), 1)[1].data.squeeze()
            correct += (pred_y == labels).sum().item()
            total += labels.size(0)

            for j in range(labels.size(0)):
                if pred_y[j] == 0 and labels[j] == 1:
                    fn += 1
                if pred_y[j] == 1 and labels[j] == 0:
                    fp += 1

    acc = correct / total

    return acc, correct, total, fn, fp


best_val_acc = 0
lowest_fn = np.inf
for epoch in range(NUM_EPOCHS):

    running_loss = 0.0
    train_correct = 0
    train_total = 0
    with tqdm(train_dataloader, unit="batch") as tepoch:
        for data in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            inputs, labels = data["image"].to(DEVICE), data["is_speech"].to(DEVICE)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            pred_y = torch.max(outputs, 1)[1].data.squeeze()
            train_correct += (pred_y == labels).sum().item()
            train_total += labels.size(0)

            tepoch.set_postfix(loss=running_loss, acc=train_correct / train_total)
            sleep(0.1)

    val_acc, _, _, fn, fp = acc_stats(net, test_dataloader)

    if val_acc > best_val_acc:
        torch.save(net.state_dict(), model_filepath)
        print(
            "Val acc improved from {:.2f} to {:.2f}. Saving model.".format(
                best_val_acc, val_acc
            )
        )
        best_val_acc = val_acc
        lowest_fn = fn

    elif val_acc == best_val_acc and fn <= lowest_fn:
        torch.save(net.state_dict(), model_filepath)
        print(
            "Val acc improved from {:.2f} to {:.2f}. Saving model.".format(
                best_val_acc, val_acc
            )
        )
        best_val_acc = val_acc
        lowest_fn = fn

    print(
        "Epoch:{}\t Loss:{:.3f}\t Train_accuracy:{:.2f}\t Val_accuracy:{:.2f}".format(
            epoch, running_loss, train_correct / train_total, val_acc
        )
    )

    lr_scheduler.step()

print("Finished Training")
