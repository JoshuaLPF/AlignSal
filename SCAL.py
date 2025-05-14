import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLossWithConv(nn.Module):
    def __init__(self, k=3, temperature=0.07, threshold=0.5):
        super(ContrastiveLossWithConv, self).__init__()
        self.k = k
        self.temperature = temperature
        self.threshold = threshold
        self.unfold = nn.Unfold(kernel_size=k, padding=k // 2)

    def forward(self, A, B):
        batch_size, channels, height, width = A.size()

        # Normalize the entire tensors A and B
        A = F.normalize(A, p=2, dim=1)
        B = F.normalize(B, p=2, dim=1)

        # Extract patches using unfold
        B_unfold = self.unfold(B).view(batch_size, channels, self.k * self.k, height, width)

        total_loss = 0.0
        count = 0

        for i in range(height):
            for j in range(width):
                # Anchor is the element (i, j) in tensor A
                anchor = A[:, :, i, j].unsqueeze(2)  # Shape: (batch_size, channels, 1)
                patches = B_unfold[:, :, :, i, j]  # Shape: (batch_size, channels, k*k)

                # Compute cosine similarity between anchor and patches
                similarities = torch.einsum('bc,bck->bk', anchor.squeeze(2), patches)  # Shape: (batch_size, k*k)

                # Generate labels based on the threshold
                labels = torch.zeros_like(similarities)
                labels[similarities > self.threshold] = 1  # Positive samples

                # Compute InfoNCE loss
                sim = similarities / self.temperature  # Shape: (batch_size, k*k)
                log_prob = F.log_softmax(sim, dim=-1)
                loss = -torch.mean(torch.sum(labels * log_prob, dim=-1))
                total_loss += loss
                count += 1

        return total_loss / count



if __name__ == "__main__":
    # Create two random tensors with shape (B, C, H, W) = (4, 1, 64, 64)
    A = torch.randn(4, 1, 12, 12).cuda()
    B = torch.randn(4, 1, 12, 12).cuda()

    # Initialize the contrastive loss function with KNN
    contrastive_loss_knn = ContrastiveLossWithConv(k=3, temperature=0.07)

    # Compute the loss
    loss = contrastive_loss_knn(A, B)
    print(f"Loss: {loss.item()}")