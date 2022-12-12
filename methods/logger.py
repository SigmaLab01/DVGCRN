import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib as mpl
import pandas as pd


class Logger():
    def __init__(self, out, name='loss', xlabel='epoch'):
        self.out = out
        self.name = name
        self.xlabel = xlabel
        self.txt_file = os.path.join(out, name + '.txt')
        self.txt_file_verified = os.path.join(out, name + '_verified.txt')
        self.plot_file = os.path.join(out, name + '.png')
        self.plot_kl_file = os.path.join(out, name + '_kl.png')
        self.plot_as_file_llh_x = os.path.join(out, name + '_anomaly_score_llhx.png')
        self.plot_as_file_llh_xz = os.path.join(out, name + '_anomaly_score_llh_xz.png')
        self.plot_as_file_llh_z = os.path.join(out, name + '_anomaly_score_llh_z.png')
        self.plot_as_file_llh_z_l = os.path.join(out, name + '_anomaly_score_llh_z')
        self.plot_as_file_llh_x_verified = os.path.join(out, name + '_anomaly_score_llh_x_verified.png')
        self.plot_as_file_llh_xz_verified = os.path.join(out, name + '_anomaly_score_llh_xz_verified.png')
        self.plot_as_file_llh_z_verified = os.path.join(out, name + '_anomaly_score_llh_z_verified.png')
        self.plot_as_file_llh_z_verified_l = os.path.join(out, name + '_anomaly_score_llh_z_verified')
        self.plot_zf_file_lt_l = os.path.join(out, name + '_zf_lt')
        self.plot_zb_file_lt_l = os.path.join(out, name + '_zb_lt')
        self.plot_zf_file_verified_l = os.path.join(out, name + '_zf_verified')
        self.plot_zb_file_verified_l = os.path.join(out, name + '_zb_verified')

    def log_trainer(self, epoch, states, t=None):
        self._print_trainer(epoch, states, t)
        self._plot(epoch, states)
        states_kl = {}
        states_kl['Epoch'] = states['Epoch']
        states_kl['KL_z'] = states['KL_z']
        self._plot_kl(epoch, states_kl)

    def log_tester(self, epoch, states, L, t=None):
        self._print_tester(epoch, states, L, t)

    def log_evaluator(self, states):
        self._print_eval(states)

    def log_evaluator_re(self, message):
        self._print_eval_result(message)

    def log_evaluator_pot(self, message):
        self._print_eval_result(message)

    def _print_trainer(self, epoch, states, t=None):
        if t is not None:
            if self.xlabel == 'epoch':
                message = '(eps: %d, time: %.5f) ' % (epoch, t)
            else:
                message = '(%s: %d, time: %.5f) ' % (self.xlabel, epoch, t)
        else:
            if self.xlabel == 'epoch':
                message = '(eps: %d) ' % (epoch)
            else:
                message = '(%s: %d) ' % (self.xlabel, epoch)
        for k, v in states.items():
            message += '%s: %.5f ' % (k, v)

        with open(self.txt_file, "a") as f:
            f.write('%s\n' % message)

    def _print_tester(self, epoch, states, L, t=None):

        message = '{},{},{},{},{}'.format(states['Last_timestamp'],
                                          states['Llh_Lt'],
                                          states['IA'],
                                          states['llh_xz_lt'],
                                          states['llh_z_lt'])

        message_verified = '{},{},{},{},{}'.format(states['Verified_timestamp'],
                                                   states['Llh_verified'],
                                                   states['IA_verified'],
                                                   states['llh_xz_verified'],
                                                   states['llh_z_verified'])
        for l in range(L - 1):
            message += ',' + str(states['llh_z_lt_{}'.format(l)])
            message_verified += ',' + str(states['llh_z_verified_{}'.format(l)])

        with open(self.txt_file, "a") as f:
            f.write('%s\n' % message)

        with open(self.txt_file_verified, "a") as f_v:
            f_v.write('%s\n' % message_verified)

        for l in range(L):
            message_zf_lt = '{}'.format(states['Last_timestamp'])
            message_zf_verified = '{}'.format(states['Verified_timestamp'])
            for idx in range(len(states['zf_lt_{}'.format(l)])):
                message_zf_lt += ',{}'.format(states['zf_lt_{}'.format(l)][idx])
                message_zf_verified += ',{}'.format(states['zf_verified_{}'.format(l)][idx])
            message_zf_lt += ',{}'.format(states['IA'])
            message_zf_verified += ',{}'.format(states['IA_verified'])

            with open(self.plot_zf_file_lt_l + '_{}.txt'.format(l), "a") as f_zf_lt:
                f_zf_lt.write('%s\n' % message_zf_lt)

            with open(self.plot_zf_file_verified_l + '_{}.txt'.format(l), "a") as f_zf_ver:
                f_zf_ver.write('%s\n' % message_zf_verified)

    def _plot_z(self, L):
        for l in range(L):
            self._plot_data(self.plot_zf_file_lt_l + '_{}.txt'.format(l), self.plot_zf_file_lt_l + '_{}.pdf'.format(l),
                            cmp_=None)
            self._plot_data(self.plot_zf_file_verified_l + '_{}.txt'.format(l),
                            self.plot_zf_file_verified_l + '_{}.pdf'.format(l), cmp_=None)

    def _print_eval(self, states):

        message = 'th:{}, p:{}, r:{}, f1score:{}, TP:{}, FN:{}, TN:{}, FP:{}, FPR:{}, TPR:{}'.format(
            states['Th'],
            states['P'],
            states['R'],
            states['F1score'],
            states['TP'],
            states['FN'],
            states['TN'],
            states['FP'],
            states['Fpr'],
            states['Tpr'])
        with open(self.txt_file, "a") as f:
            f.write('%s\n' % message)

    def _print_eval_result(self, message):
        with open(self.txt_file, "a") as f:
            f.write('%s\n' % message)

    def _plot(self, epoch, states):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(states.keys())}
        self.plot_data['X'].append(epoch)
        self.plot_data['Y'].append(
            [states[k] for k in self.plot_data['legend']])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid()
        for i, k in enumerate(self.plot_data['legend']):
            ax.plot(np.array(self.plot_data['X']),
                    np.array(self.plot_data['Y'])[:, i],
                    label=k)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.name)
        l = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        fig.savefig(self.plot_file,
                    bbox_extra_artists=(l,),
                    bbox_inches='tight')
        plt.close()

    def _plot_kl(self, epoch, states):
        if not hasattr(self, 'plot_kl'):
            self.plot_kl = {'X': [], 'Y': [], 'legend': list(states.keys())}
        self.plot_kl['X'].append(epoch)
        self.plot_kl['Y'].append(
            [states[k] for k in self.plot_kl['legend']])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid()
        for i, k in enumerate(self.plot_kl['legend']):
            ax.plot(np.array(self.plot_kl['X']),
                    np.array(self.plot_kl['Y'])[:, i],
                    label=k)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.name)
        l = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        fig.savefig(self.plot_kl_file,
                    bbox_extra_artists=(l,),
                    bbox_inches='tight')
        plt.close()

    def anomaly_score_plot_llh_x(self, fig_size=[15, 5], y_range=[-150000, 400]):
        if not os.path.exists(self.txt_file):
            raise ValueError('Unknown file: {}'.format(self.txt_file))
        timestamp_anomalyscore_label1 = np.loadtxt(self.txt_file, delimiter=',', dtype=bytes, unpack=False).astype(str)
        timestamp_anomalyscore_label2 = timestamp_anomalyscore_label1.tolist()
        timestamp_anomalyscore_label2.sort()
        ts = []
        value = []
        label = []
        for i in range(len(timestamp_anomalyscore_label2)):
            ts.append(timestamp_anomalyscore_label2[i][0])
            value.append(float(timestamp_anomalyscore_label2[i][1]))
            label.append(timestamp_anomalyscore_label2[i][2])

        timestamp = [datetime.strptime(str(t), '%Y%m%d%H%M%S') for t in ts]
        fig = plt.figure(1, figsize=(fig_size[0], fig_size[1]))
        plt.plot(timestamp, value, 'k')
        for l in range(len(label)):
            if label[l] == 'Anomaly':
                plt.vlines(timestamp[l], y_range[0], y_range[1], colors='r')
        plt.ylim(y_range[0], y_range[1])
        plt.ylabel('Anomaly Score')
        plt.gcf().autofmt_xdate()
        plt.rcParams.update({'savefig.dpi': 500})
        plt.savefig(self.plot_as_file_llh_x, bbox_inches='tight')
        plt.close()

    def anomaly_score_plot_llh_xz(self, fig_size=[15, 5], y_range=[-150000, 400]):
        if not os.path.exists(self.txt_file):
            raise ValueError('Unknown file: {}'.format(self.txt_file))
        timestamp_anomalyscore_label1 = np.loadtxt(self.txt_file, delimiter=',', dtype=bytes, unpack=False).astype(str)
        timestamp_anomalyscore_label2 = timestamp_anomalyscore_label1.tolist()
        timestamp_anomalyscore_label2.sort()
        ts = []
        value = []
        label = []
        for i in range(len(timestamp_anomalyscore_label2)):
            ts.append(timestamp_anomalyscore_label2[i][0])
            value.append(float(timestamp_anomalyscore_label2[i][3]))
            label.append(timestamp_anomalyscore_label2[i][2])

        timestamp = [datetime.strptime(str(t), '%Y%m%d%H%M%S') for t in ts]
        fig = plt.figure(1, figsize=(fig_size[0], fig_size[1]))
        plt.plot(timestamp, value, 'k')
        for l in range(len(label)):
            if label[l] == 'Anomaly':
                plt.vlines(timestamp[l], y_range[0], y_range[1], colors='r')
        plt.ylim(y_range[0], y_range[1])
        plt.ylabel('Anomaly Score')
        plt.gcf().autofmt_xdate()
        plt.rcParams.update({'savefig.dpi': 500})
        plt.savefig(self.plot_as_file_llh_xz, bbox_inches='tight')
        plt.close()

    def anomaly_score_plot_llh_z(self, fig_size=[15, 5], y_range=[-150000, 400]):
        if not os.path.exists(self.txt_file):
            raise ValueError('Unknown file: {}'.format(self.txt_file))
        timestamp_anomalyscore_label1 = np.loadtxt(self.txt_file, delimiter=',', dtype=bytes, unpack=False).astype(str)
        timestamp_anomalyscore_label2 = timestamp_anomalyscore_label1.tolist()
        timestamp_anomalyscore_label2.sort()
        ts = []
        value = []
        label = []
        for i in range(len(timestamp_anomalyscore_label2)):
            ts.append(timestamp_anomalyscore_label2[i][0])
            value.append(float(timestamp_anomalyscore_label2[i][4]))
            label.append(timestamp_anomalyscore_label2[i][2])

        timestamp = [datetime.strptime(str(t), '%Y%m%d%H%M%S') for t in ts]
        fig = plt.figure(1, figsize=(fig_size[0], fig_size[1]))
        plt.plot(timestamp, value, 'k')
        for l in range(len(label)):
            if label[l] == 'Anomaly':
                plt.vlines(timestamp[l], y_range[0], y_range[1], colors='r')
        plt.ylim(y_range[0], y_range[1])
        plt.ylabel('Anomaly Score')
        plt.gcf().autofmt_xdate()
        plt.rcParams.update({'savefig.dpi': 500})
        plt.savefig(self.plot_as_file_llh_z, bbox_inches='tight')
        plt.close()

    def anomaly_score_plot_llh_z_l(self, layer, fig_size=[15, 5], y_range=[-150000, 400]):
        if not os.path.exists(self.txt_file):
            raise ValueError('Unknown file: {}'.format(self.txt_file))
        timestamp_anomalyscore_label1 = np.loadtxt(self.txt_file, delimiter=',', dtype=bytes, unpack=False).astype(str)
        timestamp_anomalyscore_label2 = timestamp_anomalyscore_label1.tolist()
        timestamp_anomalyscore_label2.sort()
        ts = []
        value = []
        label = []
        for i in range(len(timestamp_anomalyscore_label2)):
            ts.append(timestamp_anomalyscore_label2[i][0])
            value.append(float(timestamp_anomalyscore_label2[i][5 + layer]))
            label.append(timestamp_anomalyscore_label2[i][2])

        timestamp = [datetime.strptime(str(t), '%Y%m%d%H%M%S') for t in ts]
        fig = plt.figure(1, figsize=(fig_size[0], fig_size[1]))
        plt.plot(timestamp, value, 'k')
        for l in range(len(label)):
            if label[l] == 'Anomaly':
                plt.vlines(timestamp[l], y_range[0], y_range[1], colors='r')
        plt.ylim(y_range[0], y_range[1])
        plt.ylabel('Anomaly Score')
        plt.gcf().autofmt_xdate()
        plt.rcParams.update({'savefig.dpi': 500})
        plt.savefig(self.plot_as_file_llh_z_l + '_{}.png'.format(layer), bbox_inches='tight')
        plt.close()

    def anomaly_score_plot_llh_x_verified(self, fig_size=[15, 5], y_range=[-150000, 400]):
        if not os.path.exists(self.txt_file_verified):
            raise ValueError('Unknown file: {}'.format(self.txt_file_verified))
        timestamp_anomalyscore_label1 = np.loadtxt(self.txt_file_verified, delimiter=',', dtype=bytes,
                                                   unpack=False).astype(str)
        timestamp_anomalyscore_label2 = timestamp_anomalyscore_label1.tolist()
        timestamp_anomalyscore_label2.sort()
        ts = []
        value = []
        label = []
        for i in range(len(timestamp_anomalyscore_label2)):
            ts.append(timestamp_anomalyscore_label2[i][0])
            value.append(float(timestamp_anomalyscore_label2[i][1]))
            label.append(timestamp_anomalyscore_label2[i][2])

        timestamp = [datetime.strptime(str(t), '%Y%m%d%H%M%S') for t in ts]
        fig = plt.figure(1, figsize=(fig_size[0], fig_size[1]))
        plt.plot(timestamp, value, 'k')
        for l in range(len(label)):
            if label[l] == 'Anomaly':
                plt.vlines(timestamp[l], y_range[0], y_range[1], colors='r')
        plt.ylim(y_range[0], y_range[1])
        plt.ylabel('Anomaly Score')
        plt.gcf().autofmt_xdate()
        plt.rcParams.update({'savefig.dpi': 500})
        plt.savefig(self.plot_as_file_llh_x_verified, bbox_inches='tight')
        plt.close()

    def anomaly_score_plot_llh_xz_verified(self, fig_size=[15, 5], y_range=[-150000, 400]):
        if not os.path.exists(self.txt_file_verified):
            raise ValueError('Unknown file: {}'.format(self.txt_file_verified))
        timestamp_anomalyscore_label1 = np.loadtxt(self.txt_file_verified, delimiter=',', dtype=bytes,
                                                   unpack=False).astype(str)
        timestamp_anomalyscore_label2 = timestamp_anomalyscore_label1.tolist()
        timestamp_anomalyscore_label2.sort()
        ts = []
        value = []
        label = []
        for i in range(len(timestamp_anomalyscore_label2)):
            ts.append(timestamp_anomalyscore_label2[i][0])
            value.append(float(timestamp_anomalyscore_label2[i][3]))
            label.append(timestamp_anomalyscore_label2[i][2])

        timestamp = [datetime.strptime(str(t), '%Y%m%d%H%M%S') for t in ts]
        fig = plt.figure(1, figsize=(fig_size[0], fig_size[1]))
        plt.plot(timestamp, value, 'k')
        for l in range(len(label)):
            if label[l] == 'Anomaly':
                plt.vlines(timestamp[l], y_range[0], y_range[1], colors='r')
        plt.ylim(y_range[0], y_range[1])
        plt.ylabel('Anomaly Score')
        plt.gcf().autofmt_xdate()
        plt.rcParams.update({'savefig.dpi': 500})
        plt.savefig(self.plot_as_file_llh_xz_verified, bbox_inches='tight')
        plt.close()

    def anomaly_score_plot_llh_z_verified(self, fig_size=[15, 5], y_range=[-150000, 400]):
        if not os.path.exists(self.txt_file_verified):
            raise ValueError('Unknown file: {}'.format(self.txt_file_verified))
        timestamp_anomalyscore_label1 = np.loadtxt(self.txt_file_verified, delimiter=',', dtype=bytes,
                                                   unpack=False).astype(str)
        timestamp_anomalyscore_label2 = timestamp_anomalyscore_label1.tolist()
        timestamp_anomalyscore_label2.sort()
        ts = []
        value = []
        label = []
        for i in range(len(timestamp_anomalyscore_label2)):
            ts.append(timestamp_anomalyscore_label2[i][0])
            value.append(float(timestamp_anomalyscore_label2[i][4]))
            label.append(timestamp_anomalyscore_label2[i][2])

        timestamp = [datetime.strptime(str(t), '%Y%m%d%H%M%S') for t in ts]
        fig = plt.figure(1, figsize=(fig_size[0], fig_size[1]))
        plt.plot(timestamp, value, 'k')
        for l in range(len(label)):
            if label[l] == 'Anomaly':
                plt.vlines(timestamp[l], y_range[0], y_range[1], colors='r')
        plt.ylim(y_range[0], y_range[1])
        plt.ylabel('Anomaly Score')
        plt.gcf().autofmt_xdate()
        plt.rcParams.update({'savefig.dpi': 500})
        plt.savefig(self.plot_as_file_llh_z_verified, bbox_inches='tight')
        plt.close()

    def anomaly_score_plot_llh_z_verified_l(self, layer, fig_size=[15, 5], y_range=[-15, 4]):
        if not os.path.exists(self.txt_file_verified):
            raise ValueError('Unknown file: {}'.format(self.txt_file_verified))
        timestamp_anomalyscore_label1 = np.loadtxt(self.txt_file_verified, delimiter=',', dtype=bytes,
                                                   unpack=False).astype(str)
        timestamp_anomalyscore_label2 = timestamp_anomalyscore_label1.tolist()
        timestamp_anomalyscore_label2.sort()
        ts = []
        value = []
        label = []
        for i in range(len(timestamp_anomalyscore_label2)):
            ts.append(timestamp_anomalyscore_label2[i][0])
            value.append(float(timestamp_anomalyscore_label2[i][5 + layer]))
            label.append(timestamp_anomalyscore_label2[i][2])

        timestamp = [datetime.strptime(str(t), '%Y%m%d%H%M%S') for t in ts]
        fig = plt.figure(1, figsize=(fig_size[0], fig_size[1]))
        plt.plot(timestamp, value, 'k')
        for l in range(len(label)):
            if label[l] == 'Anomaly':
                plt.vlines(timestamp[l], y_range[0], y_range[1], colors='r')
        plt.ylim(y_range[0], y_range[1])
        plt.ylabel('Anomaly Score')
        plt.gcf().autofmt_xdate()
        plt.rcParams.update({'savefig.dpi': 500})

        plt.savefig(self.plot_as_file_llh_z_verified_l + '_{}.png'.format(layer), bbox_inches='tight')
        plt.close()

    def _plot_data(self, input_file, output_file,
                   fig_size=[50, 50], box_color='silver',
                   bwith=10., cmp_=None, linewidth=2, legend=True, fontsize=20):
        data_label = pd.read_csv(input_file, header=None, index_col=0).values
        data = data_label[:, 0:-1]
        label_info = data_label[:, -1].tolist()
        df = pd.DataFrame(data)
        ax = plt.gca()
        if cmp_ is not None:
            _cmp = mpl.colors.ListedColormap(list([cmp_ for i in range(df.shape[1])]))
            ax = df.plot(subplots=True, figsize=(fig_size[0], fig_size[1]), linewidth=linewidth, legend=legend,
                         fontsize=fontsize, colormap=_cmp, xticks=[], yticks=[])
        else:
            ax = df.plot(subplots=True, figsize=(fig_size[0], fig_size[1]), linewidth=linewidth, legend=legend,
                         fontsize=fontsize, xticks=[], yticks=[])

        anomaly_idx = []
        for i in range(len(label_info)):
            if str(label_info[i]) == 'Anomaly':
                anomaly_idx.append(i)
            else:
                continue
        if len(anomaly_idx) == 0:
            anomaly_seq = []
        else:
            anomaly_seq = []
            for j in range(len(anomaly_idx)):
                if j == 0:
                    left_idx = anomaly_idx[j]
                elif 0 < j < len(anomaly_idx) - 1 and anomaly_idx[j + 1] - anomaly_idx[j] != 1:
                    right_idx = anomaly_idx[j]
                    anomaly_seq.append([left_idx, right_idx])
                    left_idx = anomaly_idx[j + 1]
                elif j == len(anomaly_idx) - 1:
                    right_idx = anomaly_idx[j]
                    anomaly_seq.append([left_idx, right_idx])

        if len(anomaly_seq) != 0:
            for span_range in anomaly_seq:
                [i.axvspan(span_range[0] - 10, span_range[1] + 10, facecolor='r', alpha=0.5) for i in ax]
        plt.savefig(output_file, bbox_inches='tight', format='pdf', edgecolor='k')
        plt.close()
