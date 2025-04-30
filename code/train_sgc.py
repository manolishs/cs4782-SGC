import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# Torch Geometric is convenient for loading the exact Planetoid splits that
# the original GCN paper used.  We can install with `pip install torch-geometric` if
# one does not have it already.
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj

from models import SGC
from preprocessing import normalize_adj, row_normalize_features, propagate_k_hops

DATASET       = "Pubmed"          # Cora | Citeseer | Pubmed
LR            = 0.2               # Learning rate (paper default)
WEIGHT_DECAY  = 5e-6              # Default from official code
EPOCHS        = 100               # “we train SGC for 100 epochs...

# Not sure which one is the best? Trying to use figure 4 in GCN paper
K_BY_DATASET = {
    "Cora": 2,
    "Citeseer": 2,
    "Pubmed": 3
}


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Same as before
    features, adj, labels, idx_train, idx_val, idx_test = load_planetoid(DATASET)
    features = row_normalize_features(features)
    adj_norm = normalize_adj(adj)

    # This time we precompute K-hops
    k = K_BY_DATASET[DATASET]
    features_k = propagate_k_hops(features, adj_norm, k).to(device)
    labels     = labels.to(device)

    model = SGC(nfeat=features_k.size(1), nclass=int(labels.max()) + 1).to(device)

    #  "The training of logistic regression
    #  is a well studied convex optimization problem and can
    #  be performed with any efficient second order method or
    #  stochastic gradient descent"
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Store metrics for plotting
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(1, EPOCHS + 1):
        # same as GCN
        model.train()
        optimizer.zero_grad()
        logits = model(features_k)
        loss   = F.cross_entropy(logits[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(features_k)
            val_loss = F.cross_entropy(logits[idx_val], labels[idx_val])
            val_acc  = accuracy(logits[idx_val], labels[idx_val])

        # Save metrics
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch:03d} | train loss {loss:.4f} | val loss {val_loss:.4f} | val acc {val_acc:.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(features_k)
        test_acc = accuracy(logits[idx_test], labels[idx_test])
        print(f"Test accuracy: {test_acc:.4f}")

    # Save plots
    os.makedirs("plots", exist_ok=True)

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss (SGC)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/sgc_loss_curve.png")
    plt.close()

    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(val_accuracies, label="Validation Accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy (SGC)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/sgc_accuracy_curve.png")
    plt.close()

    print("Saved SGC training plots to 'plots/' directory.")


if __name__ == "__main__":
    main()
