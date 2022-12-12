import numpy as np
import time, datetime
import pandas as pd
import os
import ast
import argparse
from operator import itemgetter
import pickle
from sklearn import preprocessing

class KPI_Processing():
    def __init__(self, fig_path='./'):
        self.fig_path = fig_path

    def get_label(self, input_file):
        return np.loadtxt(input_file, dtype=int)

    def kpi_concat(self, train_path, test_path, label_path, output_path, dataset):
        if not os.path.exists(train_path):
            raise ValueError('Unknown kpi datasets path: {}'.format(train_path))
        if not os.path.exists(test_path):
            raise ValueError('Unknown kpi datasets path: {}'.format(test_path))
        if not os.path.exists(label_path):
            raise ValueError('Unknown kpi datasets path: {}'.format(label_path))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if dataset == 'SMD':
            dataset_name = 'machine'
        else:
            raise ValueError('Unknown dataset: {}'.format(dataset))
        group_arr = []
        website_dict = {}
        for parent, dirnames, filenames in os.walk(train_path):
            for filename in filenames:
                group_idx = int(os.path.splitext(os.path.basename(filename))[0].split('-')[1])
                website_idx = int(os.path.splitext(os.path.basename(filename))[0].split('-')[2])

                if group_idx not in website_dict:
                    website_dict[group_idx] = [website_idx]
                else:
                    website_dict[group_idx].append(website_idx)
                if group_idx not in group_arr:  
                   group_arr.append(group_idx)
        group_arr.sort()
        for g_idx in website_dict:
            website_dict[g_idx].sort()

        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
        for g_idx in group_arr:
            for w_idx in website_dict[g_idx]:
                print('({},{})'.format(g_idx,w_idx))
                
                kpi_label_name = os.path.join(label_path, str(dataset_name)+'-' + str(g_idx) + '-'
                                              + str(w_idx) + '.txt')
                kpi_data_name_train = os.path.join(train_path, str(dataset_name) + '-' + str(g_idx) +
                                                   '-' + str(w_idx) + '.txt')
                kpi_data_name_test = os.path.join(test_path, str(dataset_name) + '-' + str(g_idx) + '-'
                                                  + str(w_idx) + '.txt')
                kpi_label_test_ = self.get_label(kpi_label_name)
                kpi_data_df_train = pd.read_csv(kpi_data_name_train, header=None)
                kpi_data_df_test = pd.read_csv(kpi_data_name_test, header=None)

                if kpi_label_test_.shape[0] < kpi_data_df_test.shape[0]:
                    for i in range(kpi_data_df_test.shape[0]-kpi_label_test_.shape[0]):
                        kpi_label_test_ = np.append(kpi_label_test_,0)
                if kpi_label_test_.shape[0] > kpi_data_df_test.shape[0]:
                    for i in range(kpi_data_df_test.shape[0]-kpi_label_test_.shape[0]):
                        kpi_label_test_ = np.delete(kpi_label_test_,-1)

                assert(kpi_data_df_test.shape[0] == kpi_label_test_.shape[0])
                kpi_label_train_ = np.zeros(kpi_data_df_train.shape[0], dtype=int)
                '''
                Normalizing each dataset, then concat them 
                '''
                kpi_data_ndarray_train = np.array(kpi_data_df_train)
                kpi_data_ndarray_test = np.array(kpi_data_df_test)
                kpi_data_ndarray_train = min_max_scaler.fit_transform(kpi_data_ndarray_train)
                kpi_data_ndarray_test = min_max_scaler.fit_transform(kpi_data_ndarray_test)
                ''' 
                if kpi_data_train is None:
                    kpi_data_train = kpi_data_ndarray_train
                else:
                    kpi_data_train = np.vstack((kpi_data_train, kpi_data_ndarray_train))
                '''
                kpi_data_test_len = kpi_data_ndarray_test.shape[0]
                kpi_label_test_len = kpi_label_test_.shape[0]
                assert(kpi_data_test_len == kpi_label_test_len)
                granularity = 1
                base_time = "20200101000000"
                time_arr = time.strptime(base_time, "%Y%m%d%H%M%S")
                base_timestamp = int(time.mktime(time_arr))
                timestamp = np.array([time.strftime("%Y%m%d%H%M%S",
                            time.localtime(base_timestamp + i*granularity*60)) for i in range(kpi_data_test_len)])
                kpi_ts_label_test = np.vstack((timestamp, kpi_label_test_))
                kpi_ts_data_T_test = np.vstack((timestamp, kpi_data_ndarray_test.T))
                kpi_ts_label_test_df = pd.DataFrame(kpi_ts_label_test, index=None)
                kpi_ts_data_T_test_df = pd.DataFrame(kpi_ts_data_T_test, index=None)

                kpi_ts_label_test_filename = os.path.join(output_path, str(dataset_name)+'-' + str(g_idx) + '-'
                                                          + str(w_idx) + '_label_test.csv')
                kpi_ts_data_test_filename = os.path.join(output_path, str(dataset_name)+'-' + str(g_idx) + '-'
                                                         + str(w_idx) + '_data_test.csv')

                kpi_ts_label_test_df.to_csv(kpi_ts_label_test_filename ,index=False, header=False)
                kpi_ts_data_T_test_df.to_csv(kpi_ts_data_test_filename ,index=False, header=False)

                kpi_data_train_len = kpi_data_ndarray_train.shape[0]
                kpi_label_train_len = kpi_label_train_.shape[0]
                assert(kpi_data_train_len == kpi_label_train_len)
                granularity = 1
                base_time = "20190101000000"
                time_arr = time.strptime(base_time, "%Y%m%d%H%M%S")
                base_timestamp = int(time.mktime(time_arr))
                timestamp = np.array([time.strftime("%Y%m%d%H%M%S",
                            time.localtime(base_timestamp + i*granularity*60)) for i in range(kpi_data_train_len)])
                kpi_ts_label_train = np.vstack((timestamp, kpi_label_train_))
                kpi_ts_data_T_train = np.vstack((timestamp, kpi_data_ndarray_train.T))
                kpi_ts_label_train_df = pd.DataFrame(kpi_ts_label_train, index=None)
                kpi_ts_data_T_train_df = pd.DataFrame(kpi_ts_data_T_train, index=None)

                kpi_ts_label_train_filename = os.path.join(output_path, str(dataset_name)+'-' + str(g_idx) + '-'
                                                           + str(w_idx) + '_label_train.csv')
                kpi_ts_data_train_filename = os.path.join(output_path, str(dataset_name)+'-' + str(g_idx) + '-'
                                                          + str(w_idx) + '_data_train.csv')

                kpi_ts_label_train_df.to_csv(kpi_ts_label_train_filename, index=False, header=False)
                kpi_ts_data_T_train_df.to_csv(kpi_ts_data_train_filename, index=False, header=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='SMD/train')
    parser.add_argument('--test_path', type=str, default='SMD/test')
    parser.add_argument('--label_path', type=str, default='SMD/test_label')
    parser.add_argument('--output_path', type=str, default='SMD_processed_1st')
    parser.add_argument('--dataset', type=str, default='SMD')
    args = parser.parse_args()

    kpi_processing = KPI_Processing('./')
    kpi_processing.kpi_concat(args.train_path, args.test_path, 
                              args.label_path, args.output_path, args.dataset)


if __name__ == '__main__':
    main()
