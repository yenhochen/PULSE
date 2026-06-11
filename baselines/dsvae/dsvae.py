import torch
import torch.nn as nn

from einops import rearrange
from collections import OrderedDict


class DSVAE(nn.Module):

    def __init__(self, config, encoder=None):

        super().__init__()
        self.config = config

        ## General parameters
        self.x_dim = config.x_dim
        self.y_dim = config.x_dim
        self.z_dim = config.z_dim
        self.v_dim = config.v_dim
        self.dropout_p = config.dropout_p
        if config.activation == "relu":
            self.activation = nn.ReLU()
        elif config.activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise SystemError("Wrong activation type!")
        self.device = config.device
        # Inference
        self.dense_x = config.dense_x
        self.dim_RNN_gv = config.dim_RNN_gv
        self.num_RNN_gv = config.num_RNN_gv
        self.dense_gv_v = config.dense_gv_v
        self.dense_xv_gxv = config.dense_xv_gxv
        self.dim_RNN_gxv = config.dim_RNN_gxv
        self.num_RNN_gxv = config.num_RNN_gxv
        self.dense_gxv_gz = config.dense_gxv_gz
        self.dim_RNN_gz = config.dim_RNN_gz
        self.num_RNN_gz = config.num_RNN_gz
        #### Generation z
        self.dim_RNN_prior = config.dim_RNN_prior
        self.num_RNN_prior = config.num_RNN_prior
        # Generation x
        self.dense_vz_x = config.dense_vz_x
        ### Beta-loss
        self.beta = config.beta
        self.encoder = encoder
        self.encoder_mode = "conv"

        self.build()

    def build(self):
        ##################
        ### Inference ####
        ##################
        ### feature x
        dic_layers = OrderedDict()
        if len(self.dense_x) == 0:
            dim_x = self.x_dim
            dic_layers["Identity"] = nn.Identity()
        else:
            dim_x = self.dense_x[-1]
            for n in range(len(self.dense_x)):
                if n == 0:
                    dic_layers["linear" + str(n)] = nn.Linear(
                        self.x_dim, self.dense_x[n]
                    )
                else:
                    dic_layers["linear" + str(n)] = nn.Linear(
                        self.dense_x[n - 1], self.dense_x[n]
                    )
                dic_layers["activation" + str(n)] = self.activation
                dic_layers["dropout" + str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_x = nn.Sequential(dic_layers)
        ### content v
        # 1. g_t^v, bi-directional recurrencce
        if self.encoder is None:
            self.encoder = nn.LSTM(
                dim_x, self.dim_RNN_gv, self.num_RNN_gv, bidirectional=True
            )
            self.encoder_mode = "rnn"

        # 2. g_t^v -> v
        dic_layers = OrderedDict()
        if len(self.dense_gv_v) == 0:  # this path
            dim_gv_v = (
                2 * self.dim_RNN_gv if self.encoder_mode == "rnn" else self.dim_RNN_gv
            )
            dic_layers["Identity"] = nn.Identity()
        else:
            dim_gv_v = self.dense_gv_v[-1]
            for n in range(len(self.dense_gv_v)):
                if n == 0:
                    dic_layers["linear" + str(n)] = nn.Linear(
                        2 * self.dim_RNN_gv, self.dense_gv_v[n]
                    )
                else:
                    dic_layers["linear" + str(n)] = nn.Linear(
                        self.dense_gv_v[n - 1], self.dense_gv_v[n]
                    )
                dic_layers["activation" + str(n)] = self.activation
                dic_layers["dropout" + str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_gv_v = nn.Sequential(dic_layers)
        self.inf_v_mean = nn.Linear(dim_gv_v, self.v_dim)
        self.inf_v_logvar = nn.Linear(dim_gv_v, self.v_dim)
        ### dynamic z
        # 1. feature_x and v -> g_t^xv
        dic_layers = OrderedDict()
        if len(self.dense_xv_gxv) == 0:  # this path
            dim_xv_gxv = dim_x + self.v_dim
            dic_layers["Identity"] = nn.Identity()
        else:
            dim_xv_gxv = self.dense_xv_gxv[-1]
            for n in range(len(self.dense_xv_gxv)):
                if n == 0:
                    dic_layers["linear" + str(n)] = nn.Linear(
                        dim_x + self.v_dim, self.dense_xv_gxv[n]
                    )
                else:
                    dic_layers["linear" + str(n)] = nn.Linear(
                        self.dense_xv_gxv[n - 1], self.dense_xv_gxv[n]
                    )
                dic_layers["activation" + str(n)] = self.activation
                dic_layers["dropout" + str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_xv_gxv = nn.Sequential(dic_layers)
        # 2. g_t^xv, bidirectional recurrence
        self.rnn_g_xv = nn.LSTM(
            dim_xv_gxv, self.dim_RNN_gxv, self.num_RNN_gxv, bidirectional=True
        )
        # 3. g_t^xv -> g_t^z
        dic_layers = OrderedDict()
        if len(self.dense_gxv_gz) == 0:  # this path
            dim_gxv_gz = 2 * self.dim_RNN_gxv
            dic_layers["Identity"] = nn.Identity()
        else:
            dim_gxv_gz = self.dense_gxv_gz[-1]
            for n in range(len(self.dense_gxv_gz)):
                if n == 0:
                    dic_layers["linear" + str(n)] = nn.Linear(
                        2 * self.dim_RNN_gxv, self.dense_gxv_gz[n]
                    )
                else:
                    dic_layers["linear" + str(n)] = nn.Linear(
                        self.dense_gxv_gz[n - 1], self.dense_gxv_gz[n]
                    )
                dic_layers["activation" + str(n)] = self.activation
                dic_layers["dropout" + str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_gxv_gz = nn.Sequential(dic_layers)
        # 4. g_t^z, forward recurrence
        self.rnn_g_z = nn.RNN(
            dim_gxv_gz, self.dim_RNN_gz, self.num_RNN_gz, bidirectional=False
        )
        # 5. g_t^z -> z_t
        self.inf_z_mean = nn.Linear(self.dim_RNN_gz, self.z_dim)
        self.inf_z_logvar = nn.Linear(self.dim_RNN_gz, self.z_dim)

        ######################
        #### generation z ####
        ######################
        self.rnn_prior = nn.LSTM(
            self.z_dim, self.dim_RNN_prior, self.num_RNN_prior, bidirectional=False
        )
        self.prior_mean = nn.Linear(self.dim_RNN_prior, self.z_dim)
        self.prior_logvar = nn.Linear(self.dim_RNN_prior, self.z_dim)

        ######################
        #### Generation x ####
        ######################
        dic_layer = OrderedDict()
        for n in range(len(self.dense_vz_x)):
            if n == 0:
                dic_layer["linear" + str(n)] = nn.Linear(
                    self.v_dim + self.z_dim, self.dense_vz_x[n]
                )
            else:
                dic_layer["linear" + str(n)] = nn.Linear(
                    self.dense_vz_x[n - 1], self.dense_vz_x[n]
                )
            dic_layer["activation" + str(n)] = self.activation
            dic_layer["dropout" + str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_vz_x = nn.Sequential(dic_layer)
        self.gen_out = nn.Linear(self.dense_vz_x[-1], self.y_dim)

    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def encoder_rnn(self, x):
        batch_size = x.shape[1]

        feature_x = self.mlp_x(x)
        _, (_v, _) = self.encoder(feature_x)
        _v = _v.view(self.num_RNN_gv, 2, batch_size, self.dim_RNN_gv)[-1, :, :, :]
        _v = torch.cat((_v[0, :, :], _v[1, :, :]), -1)
        _v = self.mlp_gv_v(_v)

        return _v, feature_x

    def encoder_conv(self, x):
        "x: t b c"
        # batch_size = x.shape[1]

        feature_x = self.mlp_x(x)
        x_ = rearrange(x, "t b c -> b t c")
        _v, _ = self.encoder(x_)
        return _v, feature_x

    def inference(self, x):
        # x: (t, b, c)
        seq_len = x.shape[0]
        batch_size = x.shape[1]
        x_dim = x.shape[2]

        # 1. Feature x # 2. Infer content v
        if self.encoder_mode == "rnn":
            _v, feature_x = self.encoder_rnn(x)
        elif self.encoder_mode == "conv":
            _v, feature_x = self.encoder_conv(x)

        context = _v.clone()

        v_mean = self.inf_v_mean(_v)
        v_logvar = self.inf_v_logvar(_v)
        v = self.reparameterization(v_mean, v_logvar)

        # 2. Infer dynamic latent representation z
        v_dim = v.shape[-1]
        v_expand = v.expand(seq_len, batch_size, v_dim)

        xv_cat = torch.cat((feature_x, v_expand), -1)
        g_xv = self.mlp_xv_gxv(xv_cat)
        g_xv, _ = self.rnn_g_xv(g_xv)
        g_xv = self.mlp_gxv_gz(g_xv)
        g_z, _ = self.rnn_g_z(g_xv)
        z_mean = self.inf_z_mean(g_z)
        z_logvar = self.inf_z_logvar(g_z)
        z = self.reparameterization(z_mean, z_logvar)

        return z, z_mean, z_logvar, v, v_mean, v_logvar, context

    def generation_z(self, z_tm1):

        z_p, _ = self.rnn_prior(z_tm1)
        z_mean_p = self.prior_mean(z_p)
        z_logvar_p = self.prior_logvar(z_p)

        return z_mean_p, z_logvar_p

    def generation_x(self, v, z):

        seq_len = z.shape[0]
        batch_size = z.shape[1]
        z_dim = z.shape[2]
        v_dim = v.shape[-1]

        # concatenate v and z_t
        v_expand = v.expand(seq_len, batch_size, v_dim)
        vz_cat = torch.cat((v_expand, z), -1)

        # mlp to output y
        y = self.gen_out(self.mlp_vz_x(vz_cat))

        return y

    def forward(self, x):

        # need input:  (seq_len, batch_size, x_dim)
        _, batch_size, _ = x.shape
        # main part
        (
            self.z,
            self.z_mean,
            self.z_logvar,
            self.v,
            self.v_mean,
            self.v_logvar,
            context,
        ) = self.inference(x)
        z_0 = torch.zeros(1, batch_size, self.z_dim).to(self.device)
        z_tm1 = torch.cat([z_0, self.z[:-1, :, :]], 0)
        self.z_mean_p, self.z_logvar_p = self.generation_z(z_tm1)
        y = self.generation_x(self.v, self.z)

        self.v_mean_p = torch.zeros_like(self.v_mean)
        self.v_logvar_p = torch.zeros_like(self.v_logvar)

        return y, context

    def get_info(self):
        info = []
        info.append("----- Inference ----")
        info.append(">>>> Feature x")
        for layer in self.mlp_x:
            info.append(layer)
        info.append(">>>> Content v")
        info.append(self.rnn_g_v)
        for layer in self.mlp_gv_v:
            info.append(layer)
        info.append(self.inf_v_mean)
        info.append(self.inf_v_logvar)
        info.append(">>>> Dynamics z")
        for layer in self.mlp_xv_gxv:
            info.append(layer)
        info.append(self.rnn_g_xv)
        for layer in self.mlp_gxv_gz:
            info.append(layer)
        info.append(self.rnn_g_z)
        info.append(self.inf_z_mean)
        info.append(self.inf_z_logvar)

        info.append("----- Generation x -----")
        for layer in self.mlp_vz_x:
            info.append(layer)
        info.append(self.gen_out)

        info.append("----- Generation z -----")
        info.append(self.rnn_prior)
        info.append(self.prior_mean)
        info.append(self.prior_logvar)

        return info
