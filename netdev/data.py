import torch
from utils import rand_select_n


class Miner():
    """ Mines data for the hardest pairs in a subset of data
        To be called in training loop.
    """
    def __init__(self, model=None, positive_ratio=0.5, device=None, objective=None):
        """
            Parameters:
                model: any callable object that produces something to put into
                    the objective function
                positive_ratio (float): ratio of postive samples with respect
                    to total
                device (str): torch device to put data onto ['cuda' | 'cpu']
                objective: any callable object that produces output for each
                    sample
        """
        assert model is not None
        assert objective is not None
        assert positive_ratio >= 0 and positive_ratio <= 1
        if device is None:
            device = 'cpu'

        self._dev = device
        self.model = model
        self.objective = objective
        self.r_p = positive_ratio
        return

    def mine(self, inputs_anc=None, labels_anc=None, inputs_pos=None, inputs_neg=None, labels_neg=None):
        batch_size = inputs_anc.shape[0]
        n_candidates = inputs_neg.shape[1]

        # The loader returns positive and negative images in
        #   shape (B, A, .../(C, H, W)), where A is the number of
        #   candidates. We want (BA x .../(C, H, W)) for the network
        squeezed_shape = [-1] + list(inputs_anc.shape[1:])

        squeezed_pos = inputs_pos.view(squeezed_shape).to(self._dev)
        squeezed_neg = inputs_neg.view(squeezed_shape).to(self._dev)

        # Mold anchor images into same shape, duplicating to compare loss
        dim_repeat = [n_candidates] + [1 for _ in range(len(self.squeezed_pos) - 1)]
        squeezed_anc = inputs_anc.reshape(squeezed_shape).repeat(*dim_repeat)

        # Random selection of negatives
        negatives = (torch.rand(batch_size) > self.r_p).to(self._dev)

        # Find out which anchors have no positives
        no_pos_samples = (inputs_pos[:, 0, ...] == -1).all(dim=2).all(dim=2).squeeze()
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
        hard_candidates = torch.where(flags.view(*([-1] + dim_repeat[1:])),
                                      squeezed_neg,
                                      squeezed_pos).to(self._dev)

        labels_hard = torch.where(negatives,
                                  labels_neg,
                                  labels_anc)

        # Evaluate loss for each anchor-candidate pair - don't need gradient
        with torch.no_grad():
            outs_hard, outs_anc = self.model(hard_candidates, squeezed_anc)
            loss_hard = self.objective(outs_hard, outs_anc, flags.view(-1).float())['total']

        # Find hardest candidate
        loss, i_hard = loss_hard.view(-1, n_candidates).max(dim=1)

        # Reshape for gather step
        i_hard = i_hard.view(*([-1, 1] + dim_repeat[1:])).repeat(*([1, 1, 1] + inputs_anc.shape[2:]))

        # B, A, (C, H, W)
        hard_candidates = hard_candidates.view(
            *([batch_size, n_candidates] + inputs_anc.shape[1:])
        )

        outs_hard = outs_hard.view(batch_size,
                                   n_candidates,
                                   outs_anc.shape[-1])

        outs_anc = outs_anc.view(batch_size, n_candidates, -1)[:, 0, :]

        # Use indices of hardest samples to collect them
        inputs_hard = torch.zeros_like(inputs_anc).unsqueeze(1)

        torch.gather(hard_candidates, 1, i_hard, out=inputs_hard)
        inputs_hard = inputs_hard.squeeze(1)

        if True:
            outputs = inputs_hard, labels_hard
        else:
            # Added to appease flake8
            out_hard = None

            torch.gather(outs_hard, 1, i_hard, out=out_hard)
            out_hard = torch.zeros(batch_size, 1, outs_anc.shape[-1])
            outputs = {'images_hard': inputs_hard,
                       'embeddings_hard': out_hard,
                       'embeddings_anc': outs_anc,
                       'labels_hard': labels_hard,
                       'loss': loss}

        return outputs

    def __call__(self, inputs_anc=None, labels_anc=None, inputs_pos=None, inputs_neg=None, labels_neg=None):
        return self.mine(inputs_anc=inputs_anc, labels_anc=labels_anc, inputs_pos=inputs_pos,
                         inputs_neg=inputs_neg, labels_neg=labels_neg)
