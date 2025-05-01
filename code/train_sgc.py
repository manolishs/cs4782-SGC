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

from models import SGC
from preprocessing import normalize_adj, row_normalize_features, propagate_k_hops

DATASET       = "Pubmed"            # Cora | Citeseer | Pubmed
K             = 2
LR            = 0.2               # Learning rate (paper default)
WEIGHT_DECAY  = 5e-6              # Default from official code
EPOCHS        = 100               # “we train SGC for 100 epochs...
RUNS          = 10                # Number of runs

# # Not sure which one is the best? Trying to use figure 4 in SGC paper
# K_BY_DATASET = {
#     "Cora": 2,
#     "Citeseer": 2,
#     "Pubmed": 3
# }


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

def run_single_training(features_k, labels, idx_train, idx_val, idx_test, device):
    model = SGC(nfeat=features_k.size(1), 
                nclass=int(labels.max()) + 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(features_k)
        loss = F.cross_entropy(logits[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(features_k)
            val_loss = F.cross_entropy(logits[idx_val], labels[idx_val])
            val_acc = accuracy(logits[idx_val], labels[idx_val])

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        val_accuracies.append(val_acc)

    model.eval()
    with torch.no_grad():
        logits = model(features_k)
        test_acc = accuracy(logits[idx_test], labels[idx_test])

    return model, train_losses, val_losses, val_accuracies, test_acc


def save_run_results(run_dir, train_losses, val_losses, val_accuracies, test_acc, model):
    # Save loss plot
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Train vs Validation Loss (SGC)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss_curve.png"))
    plt.close()

    # Save accuracy plot
    plt.figure()
    plt.plot(val_accuracies, label="Validation Accuracy", color="green")
    plt.title("Validation Accuracy (SGC)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "accuracy_curve.png"))
    plt.close()

    # Save logs
    with open(os.path.join(run_dir, "metrics.txt"), "w") as f:
        for epoch, (tl, vl, va) in enumerate(zip(train_losses, val_losses, val_accuracies), 1):
            f.write(f"Epoch {epoch}: train_loss={tl:.4f}, val_loss={vl:.4f}, val_acc={va:.4f}\n")
        f.write(f"\nFinal test accuracy: {test_acc:.4f}\n")

    # Save model
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_root = f"results/SGC/{DATASET}"
    os.makedirs(save_root, exist_ok=True)

    # Same as before
    features, adj, labels, idx_train, idx_val, idx_test = load_planetoid(DATASET)
    features = row_normalize_features(features)
    adj_norm = normalize_adj(adj)

    # This time we precompute K-hops
    features_k = propagate_k_hops(features, adj_norm, K).to(device)
    labels     = labels.to(device)

    all_train_losses = []
    all_val_losses = []
    all_val_accuracies = []
    all_test_accuracies = []

    for run in range(1, RUNS + 1):
        print(f"\n=== Run {run}/{RUNS} ===")
        run_dir = os.path.join(save_root, f"run_{run}")
        os.makedirs(run_dir, exist_ok=True)

        model, train_losses, val_losses, val_accuracies, test_acc = run_single_training(
            features_k, labels, idx_train, idx_val, idx_test, device
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