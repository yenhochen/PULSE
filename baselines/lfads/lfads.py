import torch
import torch.nn as nn

from baselines.lfads.recons import Gaussian
from baselines.lfads.encoder import LFADSConvEncoder
from baselines.lfads.decoder import Decoder
from baselines.lfads.metrics import ExpSmoothedMetric
from baselines.lfads.augmentation import AugmentationStack, CoordinatedDropout
from baselines.lfads.priors import (
    Null,
    AutoregressiveMultivariateNormal,
    MultivariateNormal,
)
from baselines.lfads.readin_readout import FanInLinear


class LFADS(nn.Module):
    def __init__(
        self,
        config,
        encoder=None,
    ):
        super().__init__()

        self.config = config

        # this is hyperspecific to continuous datasets
        self.variational = config.variational

        if self.variational:
            self.co_prior = AutoregressiveMultivariateNormal(
                tau=10.0, nvar=0.1, shape=config.co_dim
            )
            self.ic_prior = MultivariateNormal(
                mean=0, variance=0.1, shape=config.ic_dim
            )
        else:
            self.co_prior = Null()
            self.ic_prior = Null()

        self.recon = Gaussian()

        self.readin = torch.nn.Identity()
        self.readout = FanInLinear(
            in_features=config.fac_dim, out_features=config.encod_data_dim * 2
        )

        # cd = CoordinatedDropout(cd_rate=config.cd_rate, cd_pass_rate=config.cd_pass_rate, ic_enc_seq_len=config.ic_enc_seq_len)
        # self.train_aug_stack = AugmentationStack(transforms=[cd], batch_order=[0], loss_order=[0])
        self.train_aug_stack = AugmentationStack()
        self.infer_aug_stack = AugmentationStack()

        # self.config.co_prior = self.co_prior
        if not self.variational:
            assert isinstance(self.ic_prior, Null) and isinstance(self.co_prior, Null)

        self.use_con = all(
            [config.ci_enc_dim > 0, config.con_dim > 0, config.co_dim > 0]
        )

        # if encoder is None:
        # self.encoder_ = Encoder(config)
        # else:
        self.encoder_ = LFADSConvEncoder(config, encoder)  # use dilated conv encoder
        self.decoder = Decoder(config, self.co_prior)
        self.valid_recon_smth = ExpSmoothedMetric(coef=0.3)

    def forward(
        self,
        batch,
        sample_posteriors: bool = False,
        output_means: bool = True,
    ):
        """
        Forward pass through the model.

        Parameters
        ----------
        batch : Dict[int, SessionBatch]
            A dictionary of SessionBatch objects, where each key is a session index and each value is a SessionBatch object.
        sample_posteriors : bool, optional
            If True, samples from the posterior distributions, otherwise passes the mean. Default is False.
        output_means : bool, optional
            If True, converts the output parameters to means. Otherwise outputs distribution parameters. Default is True.

        Returns
        -------
        Dict[int, SessionOutput]
            A dictionary of SessionOutput objects, where each key is a session and each value is a SessionOutput object.
        """

        outputs = {
            "output_params": [],
            "factors": [],
            "ic_mean": [],
            "ic_std": [],
            "co_means": [],
            "co_stds": [],
            "gen_states": [],
            "gen_init": [],
            "gen_inputs": [],
            "con_states": [],
        }

        # estimate posterior and sample
        encod_data = self.readin(batch)
        ic_mean, ic_std, ci = self.encoder_(encod_data)
        ic_post = self.ic_prior.make_posterior(ic_mean, ic_std)
        ic_samp = ic_post.rsample() if sample_posteriors else ic_mean

        # build ext inputs
        n_samps, n_steps, _ = encod_data.shape
        ext_input = torch.zeros(n_samps, n_steps, 0).to(self.config.device)

        # decode given sampled posterior
        (
            gen_init,
            gen_states,
            con_states,
            co_means,
            co_stds,
            gen_inputs,
            factors,
        ) = self.decoder(ic_samp, ci, ext_input, sample_posteriors=sample_posteriors)

        output_params_ = self.readout(factors)
        output_params = self.recon.reshape_output_params(output_params_)
        if output_means:
            output_params = self.recon.compute_means(output_params)

        outputs["output_params"] = output_params
        outputs["factors"] = factors
        outputs["ic_mean"] = ic_mean
        outputs["ic_std"] = ic_std
        outputs["co_means"] = co_means
        outputs["co_stds"] = co_stds
        outputs["gen_states"] = gen_states
        outputs["gen_inits"] = gen_init
        outputs["gen_inputs"] = gen_inputs
        outputs["con_states"] = con_states

        return outputs
