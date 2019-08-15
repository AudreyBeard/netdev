# TODO
# [ ] change shape requirements to support shaping
import torch
from ..utils import rand_select, rand_select_n, seed_rng

class MiniBatchMiner(object):
    """ Mines data for the hardest pairs in a subset of data
        To be called in training loop.
    """
    def __init__(self, model=None, positive_ratio=0.5, torch_device=None, objective=None):
        assert model is not None
        assert objective is not None
        assert positive_ratio >= 0 and positive_ratio <= 1
        if torch_device is None:
            torch_device = 'cpu'

        self._dev = torch_device
        self.model = model
        self.objective = objective
        self.r_p = positive_ratio
        return

    def mine(self, inputs_a=None, labels_a=None, inputs_p=None, inputs_n=None, labels_n=None):
        batch_size = inputs_a.shape[0]
        n_candidates = inputs_n.shape[1]

        # The loader returns positive and negative images in
        #   shape (B, A, C, H, W), where A is the number of
        #   candidates. We want (BA x C x H x W) for the network
        squeezed_shape = [-1] + list(inputs_a.shape[1:])

        squeezed_p = inputs_p.view(squeezed_shape).to(self._dev)
        squeezed_n = inputs_n.view(squeezed_shape).to(self._dev)

        # Mold anchor images into same shape, duplicating to compare loss
        repeats = [1 for i in squeezed_shape]
        repeats[0] = n_candidates
        squeezed_a = inputs_a.reshape(squeezed_shape).repeat(repeats)

        # Random selection of negatives
        negatives = (torch.rand(batch_size) > self.r_p).to(self._dev)

        # Find out which anchors have no positives
        no_pos_samples = (inputs_p[:, 0, ...] == -1).all(dim=2).all(dim=2).squeeze()
        idx_has_pos = (no_pos_samples - 1).nonzero().squeeze()
        n_flip = min(no_pos_samples.sum().item(), len(idx_has_pos))

        # Grab negatives instead of positives
        # NOTE this assumes that at least half the samples have a positive
        # TODO redesign so this is not assumed
        if n_flip > 0:
            use_pos_instead = rand_select_n(idx_has_pos.cpu(), n_flip)
            use_neg_instead = no_pos_samples.nonzero().squeeze(0)
            negatives[use_pos_instead] = 0
            negatives[use_neg_instead] = 1

        flags = negatives.view(-1, 1).repeat(1, n_candidates)

        # Prepare candidates for hard samples
        hard_candidates = torch.where(flags.view(-1, 1, 1, 1),
                                      squeezed_n,
                                      squeezed_p).to(self._dev)

        labels_hard = torch.where(negatives,
                                  labels_n,
                                  labels_a)

        # Evaluate loss for each anchor-candidate pair - don't need gradient
        with torch.no_grad():
            outs_h, outs_a = self.model(hard_candidates, squeezed_a)
            loss_h = self.objective(outs_h, outs_a, flags.view(-1).float())['total']

        # Find hardest candidate
        loss, i_hard = loss_h.view(-1, n_candidates).max(dim=1)

        # Reshape for gather step
        i_hard = i_hard.view(-1, 1, 1, 1, 1).repeat(1, 1, 1, inputs_a.shape[2], inputs_a.shape[3])
        hard_candidates = hard_candidates.view(batch_size,
                                               n_candidates,
                                               inputs_a.shape[1],
                                               inputs_a.shape[2],
                                               inputs_a.shape[3])

        outs_h = outs_h.view(batch_size,
                             n_candidates,
                             outs_a.shape[-1])

        outs_a = outs_a.view(batch_size, n_candidates, -1)[:, 0, :]

        # Use indices of hardest samples to collect them
        inputs_hard = torch.zeros_like(inputs_a).unsqueeze(1)

        torch.gather(hard_candidates, 1, i_hard, out=inputs_hard)
        inputs_hard = inputs_hard.squeeze(1)

        if True:
            outputs = inputs_hard, labels_hard
        else:
            # Added to please flake8
            out_hard = None

            torch.gather(outs_h, 1, i_hard, out=out_hard)
            out_hard = torch.zeros(batch_size, 1, outs_a.shape[-1])
            outputs = {'inputs_h': inputs_hard,
                       'embeddings_h': out_hard,
                       'embeddings_a': outs_a,
                       'labels_h': labels_hard,
                       'loss': loss}

        return outputs

    def __call__(self, inputs_a=None, labels_a=None, inputs_p=None, inputs_n=None, labels_n=None):
        return self.mine(inputs_a=inputs_a, labels_a=labels_a, inputs_p=inputs_p,
                         inputs_n=inputs_n, labels_n=labels_n)
