import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli


def pad_mask(mask, data, value):
    """Adds padding to I/O masks for CD and SV in cases where
    reconstructed data is not the same shape as the input data.
    """
    t_forward = data.shape[1] - mask.shape[1]
    n_heldout = data.shape[2] - mask.shape[2]
    pad_shape = (0, n_heldout, 0, t_forward)
    return F.pad(mask, pad_shape, value=value)


class CoordinatedDropout:
    def __init__(self, cd_rate, cd_pass_rate, ic_enc_seq_len):
        self.cd_rate = cd_rate
        self.ic_enc_seq_len = ic_enc_seq_len
        self.cd_input_dist = Bernoulli(1 - cd_rate)
        self.cd_pass_dist = Bernoulli(cd_pass_rate)
        # Use FIFO for grad masks
        self.grad_masks = []

    def process_batch(self, batch):
        # encod_data, *other_data = batch
        encod_data = batch

        # Only use CD where we are inferring rates (none inferred for IC segment)
        unmaskable_data = encod_data[:, : self.ic_enc_seq_len, :]
        maskable_data = encod_data[:, self.ic_enc_seq_len :, :]

        # Sample a new CD mask at each training step
        device = encod_data.device
        cd_mask = self.cd_input_dist.sample(maskable_data.shape).to(device)
        pass_mask = self.cd_pass_dist.sample(maskable_data.shape).to(device)

        # Save the gradient mask for `process_outputs`
        if self.cd_rate > 0:
            grad_mask = torch.logical_or(torch.logical_not(cd_mask), pass_mask).float()
        else:
            # If cd_rate == 0, turn off CD
            grad_mask = torch.ones_like(cd_mask)
        # Store the grad_mask for later
        self.grad_masks.append(grad_mask)
        # Mask and scale post-CD input so it has the same sum as the original data
        cd_masked_data = maskable_data * cd_mask / (1 - self.cd_rate)
        # Concatenate the data from the IC encoder segment if using
        cd_input = torch.cat([unmaskable_data, cd_masked_data], axis=1)

        return cd_input
        # return cd_input, *other_data

    def process_losses(self, recon_loss, *args):
        # First-in-first-out
        grad_mask = self.grad_masks.pop(0)
        # Expand mask, but don't block gradients
        grad_mask = pad_mask(grad_mask, recon_loss, 1.0)
        # Block gradients with respect to the masked outputs
        grad_loss = recon_loss * grad_mask
        nograd_loss = (recon_loss * (1 - grad_mask)).detach()
        cd_loss = grad_loss + nograd_loss

        return cd_loss

    def reset(self):
        self.grad_masks = []


class AugmentationStack:
    def __init__(self, transforms=[], batch_order=[], loss_order=[]):
        # Build lists of input and output transformations to apply
        self.batch_transforms = [transforms[i] for i in batch_order]
        self.loss_transforms = [transforms[i] for i in loss_order]
        # Check that the transformations have the correct functions defined
        assert all([hasattr(t, "process_batch") for t in self.batch_transforms])
        assert all([hasattr(t, "process_losses") for t in self.loss_transforms])

    def process_batch(self, batch):
        for transform in self.batch_transforms:
            batch = transform.process_batch(batch)
        return batch
        # return SessionBatch(*batch)

    def process_losses(
        self,
        losses,
        batch,
        #    log_fn, data_split
    ):
        for transform in self.loss_transforms:
            losses = transform.process_losses(losses, batch)
            #   , log_fn, data_split
            #   )
        return losses

    def reset(self):
        for transform in {*self.batch_transforms, *self.loss_transforms}:
            if hasattr(transform, "reset"):
                transform.reset()
