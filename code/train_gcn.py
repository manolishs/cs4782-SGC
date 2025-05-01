import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import numpy as np
import json

# Torch Geometric is convenient for loading the exact Planetoid splits that
# the original GCN paper used.  We can install with `pip install torch-geometric` if
# one does not have it already.
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj

# your GraphConv + GCN implementation
from models import GCN                    
from preprocessing import normalize_adj, row_normalize_features

# Parameters are from the GCN paper itself
DATASET      = "Pubmed"       # Cora, Citeseer, or Pubmed
HIDDEN       = 16               # 16 units as in the paper for citation networks
LR           = 0.01             # learning rate
DROPOUT      = 0.5              # dropout probability (0.5 for citation graphs)
WEIGHT_DECAY = 5e-4             # L2 regularisation on all weights
EPOCHS       = 200              # fixed 200 training epochs
PATIENCE     = 10               # early‑stopping window on val‑loss
RUNS         = 10               # number of runs


def load_planetoid(name):
    """Returns (features, adjacency, labels, idx_train, idx_val, idx_test)."""
    dataset = Planetoid(root="data", name=name)
    data = dataset[0]

    # Planetoid gives us boolean masks for the paper's fixed splits
    idx_train = data.train_mask.nonzero(as_tuple=False).view(-1)
    idx_val = data.val_mask.nonzero(as_tuple=False).view(-1)
    idx_test = data.test_mask.nonzero(as_tuple=False).view(-1)

    # Dense adjacency is fine for Cora / Citeseer / Pubmed sizes
    adj = to_dense_adj(data.edge_index)[0]

    return data.x.float(), adj, data.y, idx_train, idx_val, idx_test


def accuracy(logits, labels):
    preds = logits.max(1)[1]
    return (preds == labels).sum().item() / labels.size(0)

def run_single_training(features, adj_norm, labels, idx_train, idx_val, idx_test, device):
    model = GCN(nfeat=features.size(1),
                nhid=HIDDEN,
                nclass=int(labels.max().item() + 1),
                dropout=DROPOUT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    patience_count = 0
    best_model = None

    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out = model(features, adj_norm)
        loss_train = F.cross_entropy(out[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(features, adj_norm)
            loss_val = F.cross_entropy(logits[idx_val], labels[idx_val])
            acc_val = accuracy(logits[idx_val], labels[idx_val])

        train_losses.append(loss_train.item())
        val_losses.append(loss_val.item())
        val_accuracies.append(acc_val)

        if loss_val < best_val_loss:
            best_val_loss = loss_val
            best_model = model.state_dict()
            patience_count = 0
        else:
            patience_count += 1
            if patience_count == PATIENCE:
                break

    model.load_state_dict(best_model)
    model.eval()
    with torch.no_grad():
        logits = model(features, adj_norm)
        test_acc = accuracy(logits[idx_test], labels[idx_test])

    return model, train_losses, val_losses, val_accuracies, test_acc


def save_run_results(run_dir, train_losses, val_losses, val_accuracies, test_acc, model):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Train vs Validation Loss (GCN)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(val_accuracies, label="Validation Accuracy", color="green")
    plt.title("Validation Accuracy (GCN)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "accuracy_curve.png"))
    plt.close()

    with open(os.path.join(run_dir, "metrics.txt"), "w") as f:
        for epoch, (tl, vl, va) in enumerate(zip(train_losses, val_losses, val_accuracies), 1):
            f.write(f"Epoch {epoch}: train_loss={tl:.4f}, val_loss={vl:.4f}, val_acc={va:.4f}\n")
        f.write(f"\nFinal test accuracy: {test_acc:.4f}\n")

    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_root = f"results/GCN/{DATASET}"
    os.makedirs(save_root, exist_ok=True)

    features, adj, labels, idx_train, idx_val, idx_test = load_planetoid(DATASET)
    features = row_normalize_features(features).to(device)
    adj_norm = normalize_adj(adj).to(device)
    labels = labels.to(device)

    all_train_losses = []
    all_val_losses = []
    all_val_accuracies = []
    all_test_accuracies = []

    for run in range(1, RUNS + 1):
        print(f"\n=== Run {run}/{RUNS} ===")
        run_dir = os.path.join(save_root, f"run_{run}")
        os.makedirs(run_dir, exist_ok=True)

        model, train_losses, val_losses, val_accuracies, test_acc = run_single_training(
            features, adj_norm, labels, idx_train, idx_val, idx_test, device
        )

        save_run_results(run_dir, train_losses, val_losses, val_accuracies, test_acc, model)

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_val_accuracies.append(val_accuracies)
        all_test_accuracies.append(test_acc)

    summary = {
        "test_accuracies": all_test_accuracies,
        "test_acc_mean": float(np.mean(all_test_accuracies)),
        "test_acc_std": float(np.std(all_test_accuracies)),
    }

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))

    with open(os.path.join(save_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nFinished training. Summary saved to:", os.path.join(save_root, "summary.json"))


if __name__ == "__main__":
    main()