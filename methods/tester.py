import torch
import os
import argparse
import torch.utils.data as data
import numpy as np
from util import KpiReader
from logger import Logger
from model import GraphStackedVRNN


def Cosine_Similarity(A, B):
    [N, D] = A.shape
    inter_product = np.matmul(A, np.transpose(B))
    len_A = np.sqrt(np.sum(A * A, axis=1, keepdims=True))
    len_B = np.sqrt(np.sum(B * B, axis=1, keepdims=True))
    len_AB = np.matmul(len_A, np.transpose(len_B))
    cos_AB = inter_product / len_AB
    cos_AB[(np.arange(N), np.arange(N))] = 1
    return cos_AB


class Tester(object):
    def __init__(self, model, device, test, testloader, log_path='log_tester', log_file='loss',
                 nsamples=None, sample_path=None, checkpoints=None):
        self.model = model
        self.model.to(device)
        self.device = device
        self.test = test
        self.testloader = testloader
        self.log_path = log_path
        self.log_file = log_file
        self.nsamples = nsamples
        self.sample_path = sample_path
        self.checkpoints = checkpoints
        self.start_epoch = 0
        self.epoch_losses = []
        self.logger = Logger(self.log_path, self.log_file)
        self.loss = {}

    def load_checkpoint(self, start_ep):
        try:
            print("Loading Chechpoint from ' {} '".format(self.checkpoints + '_epochs{}.pth'.format(start_ep)))
            checkpoint = torch.load(self.checkpoints + '_epochs{}.pth'.format(start_ep))
            self.start_epoch = checkpoint['epoch']
            self.model.beta = checkpoint['beta']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.epoch_losses = checkpoint['losses']
            print("Resuming Training From Epoch {}".format(self.start_epoch))
        except:
            print("No Checkpoint Exists At '{}', Starting Fresh Training".format(
                self.checkpoints + '_epochs{}.pth'.format(start_ep)))
            self.start_epoch = 0

    def model_test(self):
        self.model.eval()
        for i, dataitem in enumerate(self.testloader, 1):
            timestamps, labels, data = dataitem
            data = data.to(self.device)

            z_posterior_forward_list, \
            z_mean_posterior_forward_list, \
            z_logvar_posterior_forward_list, \
            z_mean_prior_forward_list, \
            z_logvar_prior_forward_list, \
            x_mu_list, \
            x_logsigma_list = self.forward_test(data)

            last_timestamp = timestamps[-1, -1, -1, -1]
            label_last_timestamp_tensor = labels[-1, -1, -1, -1]
            anomaly_index = (label_last_timestamp_tensor.numpy() == 1)
            anomaly_nums = len(label_last_timestamp_tensor.numpy()[anomaly_index])
            if anomaly_nums >= 1:
                isanomaly = "Anomaly"
            else:
                isanomaly = "Normaly"
            llh_last_timestamp = self.loglikelihood_last_timestamp(data[-1, -1, -1, :, -1],
                                                                   x_mu_list[-1][-1, -1, -1, :, -1],
                                                                   x_logsigma_list[-1][-1, -1, -1, :, -1])
            llh_z_last_timestamp = 0.0
            llh_z_last_timestamp_list = []
            z_posterior_forward_last_timestamp_list = []
            L = len(z_posterior_forward_list)
            if L > 1:
                for l in range(L - 1):
                    llh_z_last_timestamp_tmp = self.loglikelihood_last_timestamp(z_posterior_forward_list[l][-1, -1, :],
                                                                                 z_mean_prior_forward_list[L - 1 - l][
                                                                                 -1, -1, :],
                                                                                 z_logvar_prior_forward_list[L - 1 - l][
                                                                                 -1, -1, :])
                    llh_z_last_timestamp_list.append(llh_z_last_timestamp_tmp.item())
                    llh_z_last_timestamp += llh_z_last_timestamp_tmp

            for l in range(L):
                z_posterior_forward_last_timestamp = z_posterior_forward_list[l][-1, -1, :].to(
                    torch.device('cpu')).numpy().tolist()
                z_posterior_forward_last_timestamp_list.append(z_posterior_forward_last_timestamp)

            T = int(timestamps.shape[1])
            verified_t = int(T / 2)
            verified_timestamp = timestamps[-1, verified_t, -1, -1]
            label_verified_timestamp_tensor = labels[-1, verified_t, -1, -1]
            verified_anomaly_index = (label_verified_timestamp_tensor.numpy() == 1)
            verified_anomaly_nums = len(label_verified_timestamp_tensor.numpy()[verified_anomaly_index])
            if verified_anomaly_nums >= 1:
                verified_isanomaly = "Anomaly"
            else:
                verified_isanomaly = "Normaly"

            llh_verified_timestamp = self.loglikelihood_last_timestamp(data[-1, verified_t, -1, :, -1],
                                                                       x_mu_list[-1][-1, verified_t, -1, :, -1],
                                                                       x_logsigma_list[-1][-1, verified_t, -1, :, -1])

            llh_z_verified_timestamp = 0.0
            llh_z_verified_timestamp_list = []
            z_posterior_forward_verified_timestamp_list = []
            L = len(z_posterior_forward_list)
            if L > 1:
                for l in range(L - 1):
                    llh_z_verified_timestamp_tmp = self.loglikelihood_last_timestamp(
                        z_posterior_forward_list[l][-1, verified_t, :],
                        z_mean_prior_forward_list[L - 1 - l][-1, verified_t, :],
                        z_logvar_prior_forward_list[L - 1 - l][-1, verified_t, :])
                    llh_z_verified_timestamp_list.append(llh_z_verified_timestamp_tmp.item())
                    llh_z_verified_timestamp += llh_z_verified_timestamp_tmp

            for l in range(L):
                z_posterior_forward_verified_timestamp = z_posterior_forward_list[l][-1, verified_t, :].to(
                    torch.device('cpu')).numpy().tolist()
                z_posterior_forward_verified_timestamp_list.append(z_posterior_forward_verified_timestamp)
            self.loss['Last_timestamp'] = last_timestamp.item()
            self.loss['Llh_Lt'] = llh_last_timestamp.item()
            self.loss['IA'] = isanomaly
            if L > 1:
                self.loss['llh_xz_lt'] = llh_last_timestamp.item() + llh_z_last_timestamp.item()
                self.loss['llh_z_lt'] = llh_z_last_timestamp.item()
                for ly in range(L - 1):
                    self.loss['llh_z_lt_{}'.format(ly)] = llh_z_last_timestamp_list[ly]
            else:
                self.loss['llh_xz_lt'] = llh_last_timestamp.item()
                self.loss['llh_z_lt'] = 0.0
                for ly in range(L - 1):
                    self.loss['llh_z_lt_{}'.format(ly)] = 0.0
            for ly in range(L):
                self.loss['zf_lt_{}'.format(ly)] = z_posterior_forward_last_timestamp_list[ly]

            self.loss['Verified_timestamp'] = verified_timestamp.item()
            self.loss['Llh_verified'] = llh_verified_timestamp.item()
            self.loss['IA_verified'] = verified_isanomaly
            if L > 1:
                self.loss['llh_xz_verified'] = llh_verified_timestamp.item() + llh_z_verified_timestamp.item()
                self.loss['llh_z_verified'] = llh_z_verified_timestamp.item()
                for ly in range(L - 1):
                    self.loss['llh_z_verified_{}'.format(ly)] = llh_z_verified_timestamp_list[ly]
            else:
                self.loss['llh_xz_verified'] = llh_verified_timestamp.item()
                self.loss['llh_z_verified'] = 0.0
                for ly in range(L - 1):
                    self.loss['llh_z_verified_{}'.format(ly)] = 0.0
            for ly in range(L):
                self.loss['zf_verified_{}'.format(ly)] = z_posterior_forward_verified_timestamp_list[ly]
            self.logger.log_tester(self.start_epoch, self.loss, L)

        print("Testing is complete!")

    def forward_test(self, data):
        with torch.no_grad():
            z_posterior_forward_list, \
            z_mean_posterior_forward_list, \
            z_logvar_posterior_forward_list, \
            z_mean_prior_forward_list, \
            z_logvar_prior_forward_list, \
            x_mu_list, \
            x_logsigma_list, _ = self.model(data)
            return z_posterior_forward_list, \
                   z_mean_posterior_forward_list, \
                   z_logvar_posterior_forward_list, \
                   z_mean_prior_forward_list, \
                   z_logvar_prior_forward_list, \
                   x_mu_list, \
                   x_logsigma_list

    def loglikelihood_last_timestamp(self, x, recon_x_mu, recon_x_logsigma):
        llh = -0.5 * torch.sum(torch.pow(((x.float() - recon_x_mu.float()) / torch.exp(recon_x_logsigma.float())),
                                         2) + 2 * recon_x_logsigma.float() + np.log(np.pi * 2))
        return llh


def main(i, j):
    parser = argparse.ArgumentParser()
    # GPU
    parser.add_argument('--gpu_id', type=int, default=0)
    # Dataset
    parser.add_argument('--dataset_path', type=str, default='../datas/data_processed/test/machine-{}-{}'.format(i, j))
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--T', type=int, default=20)
    parser.add_argument('--win_size', type=int, default=1)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--n', type=int, default=38)
    # Model
    parser.add_argument('--layers', type=int, default=3, choices=[1, 2, 3])
    parser.add_argument('--z_dims', type=int, default=[15, 10, 5])
    parser.add_argument('--h_dims', type=int, default=[20, 15, 10])
    parser.add_argument('--emb_dim', type=int, default=256)
    # Test
    parser.add_argument('--start_epoch', type=int, default=50)
    parser.add_argument('--checkpoints_path', type=str, default='model/machine-{}-{}'.format(i, j))
    parser.add_argument('--checkpoints_file', type=str, default='')
    parser.add_argument('--checkpoints_interval', type=int, default=5)
    parser.add_argument('--log_path', type=str, default='log_tester/machine-{}-{}'.format(i, j))
    parser.add_argument('--log_file', type=str, default='')
    parser.add_argument('--nsamples', type=int, default=1)
    parser.add_argument('--sample_path', type=str, default='gen_samples')
    args = parser.parse_args()
    assert len(args.z_dims) == len(args.h_dims)
    # Set up GPU
    if torch.cuda.is_available() and args.gpu_id >= 0:
        device = torch.device('cuda:%d' % args.gpu_id)
    else:
        device = torch.device('cpu')
    # Set up paths
    if not os.path.exists(args.dataset_path):
        raise ValueError('Unknown dataset path: {}'.format(args.dataset_path))
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.exists(args.checkpoints_path):
        os.makedirs(args.checkpoints_path)
    z_dim_info = ''
    h_dim_info = ''
    for i in range(len(args.z_dims)):
        if i < len(args.z_dims) - 1:
            z_dim_info = z_dim_info + str(args.z_dims[i]) + '_'
            h_dim_info = h_dim_info + str(args.h_dims[i]) + '_'
        else:
            z_dim_info = z_dim_info + str(args.z_dims[i])
            h_dim_info = h_dim_info + str(args.h_dims[i])
    if args.checkpoints_file == '':
        args.checkpints_file = 'layers_{}_zdim{}_hdim{}_winsize{}_T{}_l{}'.format(
            args.layers,
            z_dim_info,
            h_dim_info,
            args.win_size,
            args.T,
            args.l
        )
    if args.log_file == '':
        args.log_file = 'layers_{}_zdim{}_hdim{}_winsize{}_T{}_l{}_log'.format(
            args.layers,
            z_dim_info,
            h_dim_info,
            args.win_size,
            args.T,
            args.l
        )
    # Dataloader
    kpi_value_test = KpiReader(args.dataset_path)
    test_loader = data.DataLoader(kpi_value_test,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers)
    graphstackedvrnn = GraphStackedVRNN(x_dim=args.n,
                                        z_dim=args.z_dims,
                                        h_dim=args.h_dims,
                                        emb_dim=args.emb_dim,
                                        T=args.T,
                                        w=args.win_size,
                                        n=args.n,
                                        Layers=args.layers,
                                        device=device
                                        )
    tester = Tester(graphstackedvrnn, device, kpi_value_test, test_loader,
                    log_path=args.log_path,
                    log_file=args.log_file,
                    nsamples=args.nsamples,
                    sample_path=args.sample_path,
                    checkpoints=os.path.join(args.checkpoints_path, args.checkpoints_file))
    tester.load_checkpoint(args.start_epoch)
    tester.model_test()
    tester.logger.anomaly_score_plot_llh_x(y_range=[-50, 10])
    tester.logger.anomaly_score_plot_llh_xz(y_range=[-50, 30])
    tester.logger.anomaly_score_plot_llh_z(y_range=[-50, 10])
    tester.logger.anomaly_score_plot_llh_x_verified(y_range=[-50, 10])
    tester.logger.anomaly_score_plot_llh_xz_verified(y_range=[-50, 30])
    tester.logger.anomaly_score_plot_llh_z_verified(y_range=[-50, 10])
    if args.layers > 1:
        for l in range(args.layers - 1):
            tester.logger.anomaly_score_plot_llh_z_l(l, y_range=[-30, 10])
            tester.logger.anomaly_score_plot_llh_z_verified_l(l, y_range=[-30, 10])
    tester.logger._plot_z(args.layers)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    x = [[1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8],
         [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9],
         [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9], [3, 10], [3, 11]]
    for item in x:
        main(item[0], item[1])
