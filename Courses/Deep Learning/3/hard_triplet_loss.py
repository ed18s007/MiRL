import torch
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F


class HardTripletLoss(nn.Module):
    """Hard/Hardest Triplet Loss
    (pytorch implementation of https://omoindrot.github.io/triplet-loss)

    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    """
    def __init__(self, margin=0.1, hardest=False, squared=False):
        """
        Args:
            margin: margin for triplet loss
            hardest: If true, loss is considered only hardest triplets.
            squared: If true, output is the pairwise squared euclidean distance matrix.
                If false, output is the pairwise euclidean distance matrix.
        """
        super(HardTripletLoss, self).__init__()
        self.margin = margin
        self.hardest = hardest
        self.squared = squared

    def forward(self, embeddings, labels):
        """
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)

        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        pairwise_dist = _pairwise_distance(embeddings, squared=self.squared)

        if self.hardest:
            # Get the hardest positive pairs
            mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()
            valid_positive_dist = pairwise_dist * mask_anchor_positive
            hardest_positive_dist, _ = torch.max(valid_positive_dist, dim=1, keepdim=True)

            # Get the hardest negative pairs
            mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()
            max_anchor_negative_dist, _ = torch.max(pairwise_dist, dim=1, keepdim=True)
            anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (
                    1.0 - mask_anchor_negative)
            hardest_negative_dist, _ = torch.min(anchor_negative_dist, dim=1, keepdim=True)

            # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
            triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + 0.1)
            triplet_loss = torch.mean(triplet_loss)
        else:
            anc_pos_dist = pairwise_dist.unsqueeze(dim=2)
            anc_neg_dist = pairwise_dist.unsqueeze(dim=1)

            # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
            # triplet_loss[i, j, k] will contain the triplet loss of anc=i, pos=j, neg=k
            # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
            # and the 2nd (batch_size, 1, batch_size)
            loss = anc_pos_dist - anc_neg_dist + self.margin

            mask = _get_triplet_mask(labels).float()
            triplet_loss = loss * mask

            # Remove negative losses (i.e. the easy triplets)
            triplet_loss = F.relu(triplet_loss)

            # Count number of hard triplets (where triplet_loss > 0)
            hard_triplets = torch.gt(triplet_loss, 1e-16).float()
            num_hard_triplets = torch.sum(hard_triplets)

            triplet_loss = torch.sum(triplet_loss) / (num_hard_triplets + 1e-16)

        return triplet_loss


def _pairwise_distance(x, squared=False, eps=1e-16):
    # Compute the 2D matrix of distances between all the embeddings.

    cor_mat = torch.matmul(x, x.t())
    norm_mat = cor_mat.diag()
    distances = norm_mat.unsqueeze(1) - 2 * cor_mat + norm_mat.unsqueeze(0)
    distances = F.relu(distances)

    if not squared:
        mask = torch.eq(distances, 0.0).float()
        distances = distances + mask * eps
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels):
    # Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    indices_equal = torch.eye(labels.shape[0]).to(device).byte() 
    indices_not_equal = ~indices_equal

    # Check if labels[i] == labels[j]
    label_equal = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))

    mask = indices_not_equal & label_equal

    return mask

def _get_anchor_negative_triplet_mask(labels):
    """
    To be a valid negative pair (a,n),
        - a and n are different embeddings
        - a and n have the different label
    """
    indices_equal = torch.eye(labels.size(0)).byte()
    indices_not_equal = ~indices_equal

    label_not_equal = torch.ne(labels.unsqueeze(1), labels.unsqueeze(0))

    mask = indices_not_equal & label_not_equal
    return mask


def _get_triplet_mask(labels):
    """
    To be valid, a triplet (a,p,n) has to satisfy:
        - a,p,n are distinct embeddings
        - a and p have the same label, while a and n have different label
    """
    indices_equal = torch.eye(labels.size(0)).byte()
    indices_not_equal = ~indices_equal
    i_ne_j = indices_not_equal.unsqueeze(2)
    i_ne_k = indices_not_equal.unsqueeze(1)
    j_ne_k = indices_not_equal.unsqueeze(0)
    distinct_indices = i_ne_j & i_ne_k & j_ne_k

    label_equal = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))
    i_eq_j = label_equal.unsqueeze(2)
    i_eq_k = label_equal.unsqueeze(1)
    i_ne_k = ~i_eq_k
    valid_labels = i_eq_j & i_ne_k

    mask = distinct_indices & valid_labels
    return mask


'''
class HardTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, mutual_flag = False):
        super(HardTripletLoss, self).__init__()
        self.margin = margin
        self.squared = mutual_flag

    def forward(self, inputs, targets):

        def pairwise_distances(embeddings, squared=False):
            """
            ||a-b||^2 = |a|^2 - 2*<a,b> + |b|^2
            """
            # get dot product (batch_size, batch_size)
            dot_product = embeddings.mm(embeddings.t())

            # a vector
            square_sum = dot_product.diag()

            distances = square_sum.unsqueeze(1) - 2*dot_product + square_sum.unsqueeze(0)

            distances = distances.clamp(min=0)

            if not squared:
                epsilon=1e-16
                mask = torch.eq(distances, 0).float()
                distances += mask * epsilon
                distances = torch.sqrt(distances)
                distances *= (1-mask)

            return distances

        def get_valid_positive_mask(labels):
            """
            To be a valid positive pair (a,p),
                - a and p are different embeddings
                - a and p have the same label
            """
            indices_equal = torch.eye(labels.size(0)).byte()
            indices_not_equal = ~indices_equal

            label_equal = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))

            mask = indices_not_equal & label_equal
            return mask

        def get_valid_negative_mask(labels):
            """
            To be a valid negative pair (a,n),
                - a and n are different embeddings
                - a and n have the different label
            """
            indices_equal = torch.eye(labels.size(0)).byte()
            indices_not_equal = ~indices_equal

            label_not_equal = torch.ne(labels.unsqueeze(1), labels.unsqueeze(0))

            mask = indices_not_equal & label_not_equal
            return mask


        def get_valid_triplets_mask(labels):
            """
            To be valid, a triplet (a,p,n) has to satisfy:
                - a,p,n are distinct embeddings
                - a and p have the same label, while a and n have different label
            """
            indices_equal = torch.eye(labels.size(0)).byte()
            indices_not_equal = ~indices_equal
            i_ne_j = indices_not_equal.unsqueeze(2)
            i_ne_k = indices_not_equal.unsqueeze(1)
            j_ne_k = indices_not_equal.unsqueeze(0)
            distinct_indices = i_ne_j & i_ne_k & j_ne_k

            label_equal = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))
            i_eq_j = label_equal.unsqueeze(2)
            i_eq_k = label_equal.unsqueeze(1)
            i_ne_k = ~i_eq_k
            valid_labels = i_eq_j & i_ne_k

            mask = distinct_indices & valid_labels
            return mask

        def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
            """
            get triplet loss for all valid triplets and average over those triplets whose loss is positive.
            """

            distances = pairwise_distances(embeddings, squared=squared)

            anchor_positive_dist = distances.unsqueeze(2)
            anchor_negative_dist = distances.unsqueeze(1)
            triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

            # get a 3D mask to filter out invalid triplets
            mask = get_valid_triplets_mask(labels)

            triplet_loss = triplet_loss * mask.float()
            triplet_loss.clamp_(min=0)

            # count the number of positive triplets
            epsilon = 1e-16
            num_positive_triplets = (triplet_loss > 0).float().sum()
            num_valid_triplets = mask.float().sum()
            fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + epsilon)

            triplet_loss = triplet_loss.sum() / (num_positive_triplets + epsilon)

            return triplet_loss, fraction_positive_triplets

        def batch_hard_triplet_loss(labels, embeddings, margin, squared=False):
            """
            - compute distance matrix
            - for each anchor a0, find the (a0,p0) pair with greatest distance s.t. a0 and p0 have the same label
            - for each anchor a0, find the (a0,n0) pair with smallest distance s.t. a0 and n0 have different label
            - compute triplet loss for each triplet (a0, p0, n0), average them
            """
            distances = pairwise_distances(embeddings, squared=squared)

            mask_positive = get_valid_positive_mask(labels)
            hardest_positive_dist = (distances * mask_positive.float()).max(dim=1)[0]

            mask_negative = get_valid_negative_mask(labels)
            max_negative_dist = distances.max(dim=1,keepdim=True)[0]
            distances = distances + max_negative_dist * (~mask_negative).float()
            hardest_negative_dist = distances.min(dim=1)[0]

            triplet_loss = (hardest_positive_dist - hardest_negative_dist + margin).clamp(min=0)
            triplet_loss = triplet_loss.mean()

            return triplet_loss
        return batch_hard_triplet_loss(targets, inputs, self.margin, self.squared)
'''