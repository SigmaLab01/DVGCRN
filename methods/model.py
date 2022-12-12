import torch
import torch.nn as nn
import torch.nn.functional as F
from util import LossFunctions, ReparameterizeTrick


class LinearUnit(nn.Module):
    def __init__(self, in_features, out_features, nonlinearity=nn.LeakyReLU(0.2)):
        super(LinearUnit, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, out_features),
            nonlinearity
        )

    def forward(self, x):
        return self.model(x)


class GraphModule(nn.Module):
    def __init__(self, enc_dim_list, h_dim_list, emb_dim):
        """
        :param enc_dim_list: input's dim + enc_dim
        :param h_dim_list:  h_dim[0]'s dim + h_dim
        :param emb_dim: 256 for example
        """
        super(GraphModule, self).__init__()
        self.alpha0 = nn.Parameter(0.1*torch.randn(enc_dim_list[0], emb_dim), requires_grad=True)
        self.alpha1 = nn.Parameter(0.1*torch.randn(enc_dim_list[1], emb_dim), requires_grad=True)
        self.alpha2 = nn.Parameter(0.1*torch.randn(enc_dim_list[2], emb_dim), requires_grad=True)
        self.alpha3 = nn.Parameter(0.1*torch.randn(enc_dim_list[3], emb_dim), requires_grad=True)
        # self.register_parameter('alpha0', nn.Parameter(torch.randn(enc_dim_list[0], emb_dim), requires_grad=True))
        # self.register_parameter('alpha1', nn.Parameter(torch.randn(enc_dim_list[1], emb_dim), requires_grad=True))
        # self.register_parameter('alpha2', nn.Parameter(torch.randn(enc_dim_list[2], emb_dim), requires_grad=True))
        # self.register_parameter('alpha3', nn.Parameter(torch.randn(enc_dim_list[3], emb_dim), requires_grad=True))

        self.beta0 = nn.Parameter(torch.randn(emb_dim, h_dim_list[0]), requires_grad=True)
        self.beta1 = nn.Parameter(torch.randn(emb_dim, h_dim_list[1]), requires_grad=True)
        self.beta2 = nn.Parameter(torch.randn(emb_dim, h_dim_list[2]), requires_grad=True)
        # self.register_parameter('beta0', nn.Parameter(torch.randn(emb_dim, h_dim_list[0]), requires_grad=True))
        # self.register_parameter('beta1', nn.Parameter(torch.randn(emb_dim, h_dim_list[1]), requires_grad=True))
        # self.register_parameter('beta2', nn.Parameter(torch.randn(emb_dim, h_dim_list[2]), requires_grad=True))
        # self.register_parameter('beta3', nn.Parameter(torch.randn(emb_dim, h_dim_list[3]), requires_grad=True))

        # A = torch.cat([self.alpha0, self.alpha1], 0)
        # A_cal = torch.mm(A, A.t())
        # norm
        # len_A = torch.sqrt(torch.sum(A * A, dim=1, keepdim=True))
        # A_norm = torch.mm(len_A, len_A.t())
        # A_cal = A_cal / A_norm
        # A_cal[torch.arange(A.size(0)), torch.arange(A.size(0))] = 1
        # A = F.softmax(F.relu(A_cal), dim=-1).detach().cpu().numpy()
        # plt.figure(1)
        # sns.heatmap(A, cmap='jet')
        # plt.show()

    def forward(self, l, is_infer):
        if is_infer:
            if l == 0:
                A = torch.cat([self.alpha0, self.alpha1], 0)
            elif l == 1:
                A = torch.cat([self.alpha1, self.alpha2], 0)
            else:
                A = torch.cat([self.alpha2, self.alpha3], 0)

            A_cal = torch.mm(A, A.t())
            # norm
            len_A = torch.sqrt(torch.sum(A * A, dim=1, keepdim=True))
            A_norm = torch.mm(len_A, len_A.t())
            A_cal = A_cal / A_norm
            A_cal[torch.arange(A.size(0)), torch.arange(A.size(0))] = 1
            A = F.softmax(F.relu(A_cal), dim=-1)
            return A
        else:
            if l == 3:
                W_zu = None
                W_hu = F.softmax(torch.matmul(self.alpha3, self.beta2), dim=-1)
            elif l == 2:
                W_zu = F.softmax(torch.matmul(self.alpha3, self.alpha2.t()), dim=0)
                W_hu = F.softmax(torch.matmul(self.alpha2, self.beta1), dim=-1)
            elif l == 1:
                W_zu = F.softmax(torch.matmul(self.alpha2, self.alpha1.t()), dim=0)
                W_hu = F.softmax(torch.matmul(self.alpha1, self.beta0), dim=-1)
            else:
                W_zu = F.softmax(torch.matmul(self.alpha1, self.alpha0.t()), dim=0)
                W_hu = F.softmax(torch.matmul(self.alpha0, self.beta0), dim=-1)
            return W_hu, W_zu


class EncX(nn.Module):
    """
    Extended Module in the future, such as MLP or CNN
    """
    def __init__(self):
        super(EncX, self).__init__()

    def forward(self, x):
        return x


class DecX(nn.Module):
    """
    Extended Module in the future, such as MLP or CNN
    """
    def __init__(self):
        super(DecX, self).__init__()

    def forward(self, x):
        return x


class GenerationNet(nn.Module):
    def __init__(self,
                 x_dim,
                 h_dim,
                 z_dim,
                 z_dim_upper,
                 L=0,
                 Layers=1,
                 T=20,
                 w=1,
                 n=38,
                 device=torch.device('cuda:0')
                 ):
        super(GenerationNet, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.z_upper_dim = z_dim_upper
        self.T = T
        self.w = w
        self.n = n
        self.L = L
        self.Layers = Layers

        if self.L == self.Layers-1:
            self.Pz_h_mean_forward = nn.Sequential(
                LinearUnit(self.z_dim, self.h_dim),
                LinearUnit(self.h_dim, self.z_dim)
            )
            self.Pz_h_logvar_forward = nn.Sequential(
                LinearUnit(self.h_dim, self.h_dim),
                LinearUnit(self.h_dim, self.z_dim)
            )
        else:
            self.Pz_hz_mean_forward = nn.Sequential(
                LinearUnit(2 * self.z_dim, self.h_dim),
                LinearUnit(self.h_dim, self.z_dim)
            )
            self.Pz_hz_logvar_forward = nn.Sequential(
                LinearUnit(self.h_dim + self.z_upper_dim, self.h_dim),
                LinearUnit(self.h_dim, self.z_dim)
            )

        if self.L == 0:
            self.Pz_x_mean_forward = nn.Sequential(
                LinearUnit(2 * self.x_dim, self.h_dim),
                LinearUnit(self.h_dim, self.x_dim, nonlinearity=nn.Tanh())
            )
            self.Pz_x_logvar_forward = nn.Sequential(
                LinearUnit(self.h_dim + self.z_upper_dim, self.h_dim),
                LinearUnit(self.h_dim, self.x_dim, nonlinearity=nn.Tanh())
            )

    def Pz_prior_L(self, h, W_hu):
        z_mean_prior_forward = None
        z_logvar_prior_forward = None

        for t in range(self.T):
            z_mean_prior_forward_t = F.leaky_relu(torch.matmul(h[:, t, :], W_hu.t()))
            h_t = h[:, t, :]
            z_logvar_prior_forward_t = self.Pz_h_logvar_forward(h_t)

            if z_mean_prior_forward is None:
                z_mean_prior_forward = z_mean_prior_forward_t.unsqueeze(1)
                z_logvar_prior_forward = z_logvar_prior_forward_t.unsqueeze(1)
            else:
                z_mean_prior_forward = torch.cat((z_mean_prior_forward, z_mean_prior_forward_t.unsqueeze(1)), dim=1)
                z_logvar_prior_forward = torch.cat((z_logvar_prior_forward, z_logvar_prior_forward_t.unsqueeze(1)),
                                                   dim=1)
        return z_mean_prior_forward, z_logvar_prior_forward

    def Pz_prior(self, h, z_posterior_forward_upper, W_hu, W_zu):
        z_mean_prior_forward = None
        z_logvar_prior_forward = None

        for t in range(self.T):
            h_t = torch.matmul(h[:, t, :], W_hu.t())
            z_posterior_forward_t = torch.matmul(z_posterior_forward_upper[:, t, :], W_zu)
            z_mean_prior_forward_t = F.leaky_relu(h_t + z_posterior_forward_t)
            hz_forward_t = torch.cat((h[:, t, :], z_posterior_forward_upper[:, t, :]), dim=-1)
            z_logvar_prior_forward_t = self.Pz_hz_logvar_forward(hz_forward_t)

            if z_mean_prior_forward is None:
                z_mean_prior_forward = z_mean_prior_forward_t.unsqueeze(1)
                z_logvar_prior_forward = z_logvar_prior_forward_t.unsqueeze(1)
            else:
                z_mean_prior_forward = torch.cat((z_mean_prior_forward, z_mean_prior_forward_t.unsqueeze(1)), dim=1)
                z_logvar_prior_forward = torch.cat((z_logvar_prior_forward, z_logvar_prior_forward_t.unsqueeze(1)),
                                                   dim=1)
        return z_mean_prior_forward, z_logvar_prior_forward

    def gen_Px_hz(self, h, z_posterior_forward, W_hu, W_zu):
        x_mu = None
        x_logsigma = None
        for t in range(self.T):
            h_t = torch.matmul(h[:, t, :], W_hu.t())
            z_posterior_forward_t = torch.matmul(z_posterior_forward[:, t, :], W_zu)
            x_mu_t = F.tanh(h_t + z_posterior_forward_t).view(-1, 1, 1, self.n, self.w)
            tmp = torch.cat((h[:, t, :], z_posterior_forward[:, t, :]), dim=-1)
            x_logsigma_t = self.Pz_x_logvar_forward(tmp).view(-1, 1, 1, self.n, self.w)
            if x_mu is None:
                x_mu = x_mu_t
                x_logsigma = x_logsigma_t
            else:
                x_mu = torch.cat((x_mu, x_mu_t), dim=1)
                x_logsigma = torch.cat((x_logsigma, x_logsigma_t), dim=1)
        return x_mu, x_logsigma

    def forward(self, h, z_posterior_forward_0, z_posterior_forward_upper, W_hu, W_zu):
        if self.L == 0 and self.Layers == 1:
            z_mean_prior_forward, z_logvar_prior_forward = self.Pz_prior_L(h, W_hu)
            x_mu, x_logsigma = self.gen_px_hz(h, z_posterior_forward_0, W_hu, W_zu)

        elif self.L == 0 and self.Layers != 1:
            z_mean_prior_forward, z_logvar_prior_forward = None, None
            x_mu, x_logsigma = self.gen_Px_hz(h, z_posterior_forward_0, W_hu, W_zu)
        elif self.L == self.Layers - 1 and self.Layers != 1:
            z_mean_prior_forward, z_logvar_prior_forward = self.Pz_prior_L(h, W_hu)
            x_mu, x_logsigma = None, None
        else:
            z_mean_prior_forward, z_logvar_prior_forward = self.Pz_prior(h,
                                                                         z_posterior_forward_upper,
                                                                         W_hu, W_zu)
            x_mu, x_logsigma = None, None

        return z_mean_prior_forward, z_logvar_prior_forward, x_mu, x_logsigma


class InferenceNet(nn.Module):
    def __init__(self,
                 x_dim,
                 z_dim,
                 h_dim,
                 z_dim_lower,
                 h_dim_lower,
                 L=0,
                 Layers=1,
                 T=20,
                 w=1,
                 n=38,
                 device=torch.device('cuda:0')):
        super(InferenceNet, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.z_dim_lower = z_dim_lower
        self.h_dim_lower = h_dim_lower
        self.T = T
        self.w = w
        self.n = n
        self.device = device
        self.L = L
        self.Layers = Layers
        self.rt = ReparameterizeTrick()

        if self.L == 0:
            self.phi_xz_forward = LinearUnit(self.x_dim + self.z_dim, 2 * self.h_dim)
            self.rnn_enc_forward = nn.LSTMCell(2 * self.h_dim, self.h_dim, bias=True)
        else:
            self.phi_hz_forward = LinearUnit(self.z_dim_lower + self.z_dim, 2*self.h_dim)
            self.rnn_enc_forward = nn.LSTMCell(2 * self.h_dim + self.h_dim_lower, self.h_dim, bias=True)

        self.Pz_xh_mean_forward = nn.Sequential(
            LinearUnit(self.x_dim + self.h_dim, self.h_dim),
            LinearUnit(self.h_dim, self.z_dim)
        )
        self.Pz_xh_logvar_forward = nn.Sequential(
            LinearUnit(self.x_dim + self.h_dim, self.h_dim),
            LinearUnit(self.h_dim, self.z_dim)
        )

    def infer_qz_x(self, x, h_in, z_in, batch_size, A):
        h_out = None
        h_lower_out = None

        z_posterior_forward = None
        z_mean_posterior_forward = None
        z_logvar_posterior_forward = None

        h_t = torch.zeros(batch_size, self.h_dim, device=self.device)
        c_t = torch.zeros(batch_size, self.h_dim, device=self.device)

        for t in range(self.T):
            x_h_t = torch.cat((x[:, t, :], h_t), dim=1)
            z_mean_posterior_forward_t = self.Pz_xh_mean_forward(x_h_t)
            z_logvar_posterior_forward_t = self.Pz_xh_logvar_forward(x_h_t)
            z_posterior_forward_t = self.rt.reparameterize_gaussian(z_mean_posterior_forward_t,
                                                                    z_logvar_posterior_forward_t, self.training)

            if z_posterior_forward is None:
                z_posterior_forward = z_posterior_forward_t.unsqueeze(1)
                z_mean_posterior_forward = z_mean_posterior_forward_t.unsqueeze(1)
                z_logvar_posterior_forward = z_logvar_posterior_forward_t.unsqueeze(1)
                h_out = h_t.unsqueeze(1)
            else:
                z_posterior_forward = torch.cat((z_posterior_forward, z_posterior_forward_t.unsqueeze(1)), dim=1)
                z_mean_posterior_forward = torch.cat(
                    (z_mean_posterior_forward, z_mean_posterior_forward_t.unsqueeze(1)), dim=1)
                z_logvar_posterior_forward = torch.cat(
                    (z_logvar_posterior_forward, z_logvar_posterior_forward_t.unsqueeze(1)), dim=1)
                h_out = torch.cat((h_out, h_t.unsqueeze(1)), dim=1)

            if self.L == 0:  # Current layer is the first layer
                x_z_posterior_forward_t = torch.cat((x[:, t, :], z_posterior_forward_t.view(-1, self.z_dim)), dim=1)
                phi_x_z_posterior_forward_t = torch.log(1 +
                                                        torch.exp(self.phi_xz_forward(
                                                            torch.matmul(A, x_z_posterior_forward_t.t()).t())))
                h_t, c_t = self.rnn_enc_forward(phi_x_z_posterior_forward_t, (h_t, c_t))
            else:  # Current layer is not the 1st layer
                h_z_posterior_forward_t = torch.cat((z_in[:, t, :],
                                                     z_posterior_forward_t.view(-1, self.z_dim)), dim=1)
                phi_h_z_posterior_forward_t = torch.log(1 +
                                                        torch.exp(self.phi_hz_forward(
                                                            torch.matmul(A, h_z_posterior_forward_t.t()).t())))
                phi_h_z_posterior_forward_t = torch.cat([phi_h_z_posterior_forward_t, h_in[:, t, :]], dim=1)
                h_t, c_t = self.rnn_enc_forward(phi_h_z_posterior_forward_t, (h_t, c_t))

            if h_lower_out is None:
                h_lower_out = h_t.unsqueeze(1)
            else:
                h_lower_out = torch.cat((h_lower_out, h_t.unsqueeze(1)), dim=1)

        return z_posterior_forward, z_mean_posterior_forward, z_logvar_posterior_forward, h_out, h_lower_out

    def forward(self, x, h_in, z_in, A):
        x = x.squeeze(2).squeeze(3).float()
        z_posterior_forward, z_mean_posterior_forward, z_logvar_posterior_forward, \
        h_out, h_lower_out = self.infer_qz_x(x, h_in, z_in, x.size(0), A)

        return z_posterior_forward, z_mean_posterior_forward, z_logvar_posterior_forward, \
               h_out, h_lower_out


class GraphStackedVRNN(nn.Module):
    def __init__(self,
                 x_dim = 38,
                 z_dim=[24, 12, 6],
                 h_dim = [24, 12, 6],
                 emb_dim=256,
                 T=20,
                 w=1,
                 n=36,
                 beta=0.01,
                 max_beta=1.0,
                 anneal_rate=0.1,
                 Layers=3,
                 nonlinearity=None,
                 device=torch.device('cuda:0')
                 ):
        super(GraphStackedVRNN, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.emb_dim = emb_dim
        self.T = T
        self.w = w
        self.n = n
        self.beta = beta
        self.max_beta = max_beta
        self.anneal_rate = anneal_rate
        self.Layers = Layers
        self.device = device
        self.loss_func = LossFunctions()
        self.nonlinearity = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity

        assert len(self.z_dim) == self.Layers
        assert len(self.h_dim) == self.Layers

        self.inference = nn.ModuleList()
        self.generation = nn.ModuleList()
        enc_dim_list = [self.x_dim] + self.z_dim
        h_dim_list = self.h_dim
        self.graph = GraphModule(enc_dim_list, h_dim_list, self.emb_dim)
        self.predict = nn.Sequential(LinearUnit(sum(self.h_dim), x_dim),
                                     LinearUnit(x_dim, x_dim, nonlinearity=nn.Tanh()))

        for l in range(self.Layers):
            _z_dim = self.z_dim[l]
            _h_dim = self.h_dim[l]
            if l >= 1:
                _h_dim_lower = self.h_dim[l-1]
            else:
                _h_dim_lower = 0
            if l < self.Layers-1:
                _z_dim_upper = self.z_dim[l+1]
            else:
                _z_dim_upper = 0
            if l >= 1:
                _z_dim_lower = self.z_dim[l-1]
            else:
                _z_dim_lower = 0
            self.inference.append(InferenceNet(x_dim, _z_dim, _h_dim, _z_dim_lower, _h_dim_lower,
                                               l, self.Layers, self.T, self.w, self.n, self.device))

        self._z_dim = [x_dim] + self.z_dim
        self._h_dim = [self.h_dim[0]] + self.h_dim
        for l in range(self.Layers + 1):
            _z_dim = self._z_dim[l]
            _h_dim = self._h_dim[l]
            if l == 3:
                _z_dim_upper = 0
            elif l == 2:
                _z_dim_upper = self._z_dim[3]
            elif l == 1:
                _z_dim_upper = self._z_dim[2]
            elif l == 0:
                _z_dim_upper = self._z_dim[1]
            else:
                _z_dim_upper = 0
            self.generation.append(GenerationNet(x_dim, _h_dim, _z_dim, _z_dim_upper,
                                                 l, self.Layers+1, self.T, self.w, self.n, self.device))

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias.data is not None:
                    nn.init.constant_(m.bias, 0)

    def loss_LLH(self, x, x_mu, x_logsigma):

        loglikelihood = self.loss_func.log_normal(x.float(), x_mu.float(), torch.pow(torch.exp(x_logsigma.float()), 2))
        return loglikelihood

    def loss_KL(self, z_mean_posterior_forward, z_logvar_posterior_forward,
                z_mean_prior_forward, z_logvar_prior_forward):
        z_var_posterior_forward = torch.exp(z_logvar_posterior_forward)
        z_var_prior_forward = torch.exp(z_logvar_prior_forward)
        kld_z_forward = 0.5 * torch.sum(z_logvar_prior_forward - z_logvar_posterior_forward +
                                        ((z_var_posterior_forward + torch.pow(z_mean_posterior_forward -
                                                                              z_mean_prior_forward, 2)) /
                                         z_var_prior_forward)-1)
        return kld_z_forward

    def forward(self, x):
        z_posterior_forward_list = []
        z_mean_posterior_forward_list = []
        z_logvar_posterior_forward_list = []
        h_out_list = []
        h_lower_out_list = []
        h_out_last_list = []
        for l in range(self.Layers):
            A_matrix = self.graph(l, True)
            if l == 0:
                z_posterior_forward, z_mean_posterior_forward, z_logvar_posterior_forward, \
                h_out, h_lower_out = self.inference[l](x, None, None, A_matrix)
            else:
                z_posterior_forward, z_mean_posterior_forward, z_logvar_posterior_forward, \
                h_out, h_lower_out = self.inference[l](x, h_lower_out_list[l - 1],
                                                       z_posterior_forward_list[l - 1], A_matrix)
            z_posterior_forward_list.append(z_posterior_forward)
            z_mean_posterior_forward_list.append(z_mean_posterior_forward)
            z_logvar_posterior_forward_list.append(z_logvar_posterior_forward)
            h_out_list.append(h_out)
            h_lower_out_list.append(h_lower_out)
            h_out_last_list.append(h_out[:, self.T-2, :])
        h_out_last = torch.cat(h_out_last_list, dim=-1)
        tmp_h_our_list = []
        for i in range(self.Layers + 1):
            if i == 0:
                tmp_h_our_list.append(h_out_list[0])
            else:
                tmp_h_our_list.append(h_out_list[i - 1])
        z_mean_prior_forward_list = []
        z_logvar_prior_forward_list = []
        x_mu_list = []
        x_logsigma_list = []
        for l in range(self.Layers, -1, -1):
            W_hu, W_zu = self.graph(l, False)
            if l == self.Layers and self.Layers == 1:

                z_mean_prior_forward, z_logvar_prior_forward, \
                x_mu, x_logsigma = self.generation[l](h_out_list[l],
                                                      z_posterior_forward_list[l],
                                                      None, W_hu, W_zu)
            elif l == self.Layers and self.Layers != 1:
                z_mean_prior_forward, z_logvar_prior_forward, \
                x_mu, x_logsigma = self.generation[l](tmp_h_our_list[l], None, None, W_hu, W_zu)
            elif l == 0 and self.Layers != 1:
                z_mean_prior_forward, z_logvar_prior_forward, \
                x_mu, x_logsigma = self.generation[l](tmp_h_our_list[l],
                                                      z_posterior_forward_list[l],
                                                      z_posterior_forward_list[l + 1],
                                                      W_hu, W_zu)
            else:
                z_mean_prior_forward, z_logvar_prior_forward, \
                x_mu, x_logsigma = self.generation[l](tmp_h_our_list[l],
                                                      z_posterior_forward_list[l - 1],
                                                      z_posterior_forward_list[l],
                                                      W_hu, W_zu)

            z_mean_prior_forward_list.append(z_mean_prior_forward)
            z_logvar_prior_forward_list.append(z_logvar_prior_forward)
            x_mu_list.append(x_mu)
            x_logsigma_list.append(x_logsigma)

        # predict
        x_predict = self.predict(h_out_last).view(x.size(0), 1, 1, self.x_dim, 1)

        return z_posterior_forward_list, \
               z_mean_posterior_forward_list, \
               z_logvar_posterior_forward_list, \
               z_mean_prior_forward_list, \
               z_logvar_prior_forward_list, \
               x_mu_list, \
               x_logsigma_list, x_predict
