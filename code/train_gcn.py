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

# your GraphConv + GCN implementation
from models import GCN                    
from preprocessing import normalize_adj, row_normalize_features

# Parameters are from the GCN paper itself
DATASET = "Citeseer"       # Cora, Citeseer, or Pubmed
HIDDEN = 16            # 16 units as in the paper for citation networks
LR = 0.01              # learning rate
DROPOUT = 0.5          # dropout probability (0.5 for citation graphs)
WEIGHT_DECAY = 5e-4    # L2 regularisation on all weights
EPOCHS = 200           # fixed 200 training epochs
PATIENCE = 10          # early‑stopping window on val‑loss


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
    # Do we need to do this? I am sure CPU will do just fine but this allows us to use GPU in colab
    # if anything
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset, we have our features and adj from dataset
    features, adj, labels, idx_train, idx_val, idx_test = load_planetoid(DATASET)

    # "We initialize weights using the initialization described in Glorot & Bengio (2010) and
    #  accordingly (row-)normalize input feature vectors"
    features = row_normalize_features(features).to(device)
    adj_norm = normalize_adj(adj).to(device)
    labels = labels.to(device)

    model = GCN(nfeat=features.size(1),
                nhid=HIDDEN,
                nclass=int(labels.max().item() + 1),
                dropout=DROPOUT).to(device)
    
    # “We train all models for a maximum of 200 epochs … 
    # using Adam (Kingma & Ba, 2015) with a learning rate of 0.01 …”
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    patience_count = 0

    # We'll store metrics for plotting later
    train_losses = []
    val_losses = []
    val_accuracies = []

    # Train loop
    for epoch in range(1, EPOCHS + 1):
        # Set to train
        model.train()

        # Zero out gradients
        optimizer.zero_grad()

        # Forward
        out = model(features, adj_norm)

        # loss
        loss_train = F.cross_entropy(out[idx_train], labels[idx_train])

        # Backprop
        loss_train.backward()

        # Take the step
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            logits = model(features, adj_norm)
            # Cross entropy has softmax in it, thanks NLP
            loss_val = F.cross_entropy(logits[idx_val], labels[idx_val])
            acc_val = accuracy(logits[idx_val], labels[idx_val])

        # Track metrics
        train_losses.append(loss_train.item())
        val_losses.append(loss_val.item())
        val_accuracies.append(acc_val)

        print(f"Epoch {epoch:03d}  |  train loss {loss_train:.4f}  |  val loss {loss_val:.4f}  |  val acc {acc_val:.4f}")

        # Early stopping 
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            torch.save(model.state_dict(), "best_model.pt")
            patience_count = 0
        else:
            patience_count += 1
            if patience_count == PATIENCE:
                print("Early stopping triggered")
                break

    # Test GCN
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()
    with torch.no_grad():
        logits = model(features, adj_norm)
        test_acc = accuracy(logits[idx_test], labels[idx_test])

        # As of right now our test accuracy is 0.8180 and the paper achieved 81.5!
        print(f"Test accuracy: {test_acc:.4f}")

    # Save plots
    os.makedirs("plots", exist_ok=True)

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/loss_curve.png")
    plt.close()

    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(val_accuracies, label="Validation Accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/accuracy_curve.png")
    plt.close()

    print("Saved training plots to 'plots/' directory.")


if __name__ == "__main__":
    main()
