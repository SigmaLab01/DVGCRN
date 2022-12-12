import torch
import os
import argparse
import torch.utils.data as data
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from tqdm import *
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


class Trainer(object):
    def __init__(self, model, train, trainloader, log_path='log_trainer',
                 log_file='loss', epochs=20, batch_size=1024, learning_rate=0.001,
                 checkpoints='kpi_model.path', checkpoints_interval=1, device=torch.device('cuda:0')):
        self.trainloader = trainloader
        self.train = train
        self.log_path = log_path
        self.log_file = log_file
        self.start_epoch = 0
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size
        self.model = model
        self.model.to(device)
        self.learning_rate = learning_rate
        self.checkpoints = checkpoints
        self.checkpoints_interval = checkpoints_interval
        print('Model parameters: {}'.format(self.model.parameters()))
        self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate)
        self.mse = torch.nn.MSELoss()
        self.epoch_losses = []
        self.loss = {}
        self.logger = Logger(self.log_path, self.log_file)

    def save_checkpoint(self, epoch):
        torch.save({'epoch': epoch + 1,
                    'beta': self.model.beta,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'losses': self.epoch_losses},
                   self.checkpoints + '_epochs{}.pth'.format(epoch + 1))

    def load_checkpoint(self, start_ep):
        try:
            print("Loading Chechpoint from ' {} '".format(self.checkpoints + '_epochs{}.pth'.format(start_ep)))
            checkpoint = torch.load(self.checkpoints + '_epochs{}.pth'.format(start_ep))
            self.start_epoch = checkpoint['epoch']
            self.model.beta = checkpoint['beta']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch_losses = checkpoint['losses']
            print("Resuming Training From Epoch {}".format(self.start_epoch))
        except:
            print("No Checkpoint Exists At '{}', Starting Fresh Training".format(self.checkpoints))
            self.start_epoch = 0

    def train_model(self):
        self.model.train()
        for epoch in range(self.start_epoch, self.epochs):
            losses = []
            llhs = []
            kld_zs = []
            pred = []
            graph = []
            print("Running Epoch : {}".format(epoch + 1))
            for i, dataitem in tqdm(enumerate(self.trainloader, 1)):
                _, _, data = dataitem
                batch_size = data.size(0)
                data = data.to(self.device)
                self.optimizer.zero_grad()

                z_posterior_forward_list, \
                z_mean_posterior_forward_list, \
                z_logvar_posterior_forward_list, \
                z_mean_prior_forward_list, \
                z_logvar_prior_forward_list, \
                x_mu_list, \
                x_logsigma_list, x_predict = self.model(data)

                mse = self.mse(x_predict.squeeze(), data[:, data.size(1)-1, :].squeeze()).float()
                llh = self.model.loss_LLH(data, x_mu_list[-1], x_logsigma_list[-1]) / batch_size
                kld_z = 0
                L = len(z_posterior_forward_list)
                for l in range(L):
                    kld_z += self.model.loss_KL(z_mean_posterior_forward_list[l],
                                                z_logvar_posterior_forward_list[l],
                                                z_mean_prior_forward_list[L - 1 - l],
                                                z_logvar_prior_forward_list[L - 1 - l]) / batch_size

                loss = -llh + self.model.beta * kld_z + mse.long()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                llhs.append(llh.item())
                kld_zs.append(kld_z.item())
                pred.append(mse.item())
            meanloss = np.mean(losses)
            meanllh = np.mean(llhs)
            meanz = np.mean(kld_zs)
            meanpred = np.mean(pred)
            self.epoch_losses.append(meanloss)
            print("Epoch {} : Average Loss: {} Loglikelihood: {} KL of z: {}, Beta: {} Prediction".format(
                epoch + 1, meanloss, meanllh, meanz, self.model.beta, meanpred))
            self.loss['Epoch'] = epoch + 1
            self.loss['Avg_loss'] = meanloss
            self.loss['Llh'] = meanllh
            self.loss['KL_z'] = meanz
            self.logger.log_trainer(epoch + 1, self.loss)
            if (self.checkpoints_interval > 0
                    and (epoch + 1) % self.checkpoints_interval == 0):
                self.save_checkpoint(epoch)

            if (epoch + 1) % 1 == 0:
                self.model.beta = np.minimum((self.model.beta + 0.01) * np.exp(self.model.anneal_rate * (epoch + 1)),
                                             self.model.max_beta)

        print("Training is complete!")


def main(i, j):
    parser = argparse.ArgumentParser()
    # GPU
    parser.add_argument('--gpu_id', type=int, default=0)
    # Dataset
    parser.add_argument('--dataset_path', type=str, default='../datas/data_processed/train/machine-{}-{}'.format(i, j))
    parser.add_argument('--batch_size', type=int, default=512)
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
    # Training
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--max_beta', type=float, default=1.0)
    parser.add_argument('--anneal_rate', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--start_epoch', type=int, default=50)
    parser.add_argument('--checkpoints_path', type=str, default='model/machine-{}-{}'.format(i, j))
    parser.add_argument('--checkpoints_file', type=str, default='')
    parser.add_argument('--checkpoints_interval', type=int, default=5)
    parser.add_argument('--log_path', type=str, default='log_trainer/machine-{}-{}'.format(i, j))
    parser.add_argument('--log_file', type=str, default='')

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
    kpi_value_train = KpiReader(args.dataset_path)
    train_loader = data.DataLoader(kpi_value_train,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers)
    graphstackedvrnn = GraphStackedVRNN(x_dim=args.n,
                                        z_dim=args.z_dims,
                                        h_dim=args.h_dims,
                                        emb_dim=args.emb_dim,
                                        T=args.T,
                                        w=args.win_size,
                                        n=args.n,
                                        beta=args.beta,
                                        max_beta=args.max_beta,
                                        anneal_rate=args.anneal_rate,
                                        Layers=args.layers,
                                        device=device
                                        )
    trainer = Trainer(graphstackedvrnn, kpi_value_train, train_loader,
                      log_path=args.log_path,
                      log_file=args.log_file,
                      batch_size=args.batch_size,
                      epochs=args.epochs,
                      learning_rate=args.learning_rate,
                      checkpoints=os.path.join(args.checkpoints_path, args.checkpoints_file),
                      checkpoints_interval=args.checkpoints_interval, device=device)
    trainer.load_checkpoint(args.start_epoch)
    trainer.train_model()


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    x = [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8],
         [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9],
         [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9], [3, 10], [3, 11]]
    for item in x:
        main(item[0], item[1])