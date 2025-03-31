import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os


def train_model(model, X_train, y_train, X_test, y_test,
                epochs=10, batch_size=32, lr=0.01, seed=42, weight_decay=0.0,
                log_dir="runs/default", experiment="1"):

    print(f"Запущен эксперимент № {experiment}")

    # 💻 Автоматический выбор устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")
    model = model.to(device)

    torch.manual_seed(seed)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Переводим данные в torch тензоры и на нужное устройство
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=True
    )

    writer = SummaryWriter(log_dir=log_dir)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    best_loss = float("inf")
    best_epoch = 0
    best_auc = 0.0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb).squeeze()
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            logits_train = model(X_train_tensor).squeeze()
            logits_test = model(X_test_tensor).squeeze()

            train_pred = torch.sigmoid(logits_train)
            test_pred = torch.sigmoid(logits_test)

            train_auc = roc_auc_score(y_train, train_pred.cpu())
            test_auc = roc_auc_score(y_test, test_pred.cpu())

            train_loss = criterion(logits_train, y_train_tensor).item()
            test_loss = criterion(logits_test, y_test_tensor).item()

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train AUC: {train_auc:.4f} | Test AUC: {test_auc:.4f}")

        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch + 1
            best_auc = test_auc

            torch.save(model.state_dict(), f"checkpoints/best_model_exp_{experiment}.pt")
            print(f"!!! Модель сохранена (epoch {best_epoch}, test loss = {best_loss:.4f}, AUC = {best_auc:.4f})")

        writer.add_scalars("Loss", {
            "Train": train_loss,
            "Test": test_loss
        }, epoch)

        writer.add_scalar("AUC/train", train_auc, epoch)
        writer.add_scalar("AUC/test", test_auc, epoch)

    writer.close()

    # 🎨 ROC-кривая (по лучшей эпохе)
    fpr, tpr, thresholds = roc_curve(y_test, test_pred.cpu())

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {test_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (Best Test Epoch: {best_epoch})")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f"plots/roc_curve_exp_{experiment}.png")

    return best_auc
