from enum import IntEnum
from os import walk

import numpy as np

from utils_libs import *
import pickle



class DatasetObject:
    def __init__(self, dataset, n_client, rule, unbalanced_sgm=0, rule_arg=''):
        self.dataset = dataset
        self.n_client = n_client
        self.rule = rule
        self.rule_arg = rule_arg
        rule_arg_str = rule_arg if isinstance(rule_arg, str) else '%.3f' % rule_arg
        self.name = "%s_%d_%s_%s" % (self.dataset, self.n_client, self.rule, rule_arg_str)
        self.name += '_%f' % unbalanced_sgm if unbalanced_sgm != 0 else ''
        self.unbalanced_sgm = unbalanced_sgm
        self.data_path = 'Data'
        self.set_data()

    def set_data(self):
        # Prepare data if not ready
        if not os.path.exists('%s/%s' % (self.data_path, self.name)):
            # Get Raw data                
            if self.dataset == 'mnist':
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                trnset = torchvision.datasets.MNIST(root='%s/Raw' % self.data_path,
                                                    train=True, download=True, transform=transform)
                tstset = torchvision.datasets.MNIST(root='%s/Raw' % self.data_path,
                                                    train=False, download=True, transform=transform)

                trn_load = torch.utils.data.DataLoader(trnset, batch_size=60000, shuffle=False, num_workers=1)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 1;
                self.width = 28;
                self.height = 28;
                self.n_cls = 10;

            if self.dataset == 'CIFAR10':
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                                                     std=[0.247, 0.243, 0.262])])

                trnset = torchvision.datasets.CIFAR10(root='%s/Raw' % self.data_path,
                                                      train=True, download=True, transform=transform)
                tstset = torchvision.datasets.CIFAR10(root='%s/Raw' % self.data_path,
                                                      train=False, download=True, transform=transform)

                trn_load = torch.utils.data.DataLoader(trnset, batch_size=50000, shuffle=False, num_workers=1)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 3;
                self.width = 32;
                self.height = 32;
                self.n_cls = 10;

            if self.dataset == 'CIFAR100':
                print(self.dataset)
                # mean and std are validated here: https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                                     std=[0.2675, 0.2565, 0.2761])])
                trnset = torchvision.datasets.CIFAR100(root='%s/Raw' % self.data_path,
                                                       train=True, download=True, transform=transform)
                tstset = torchvision.datasets.CIFAR100(root='%s/Raw' % self.data_path,
                                                       train=False, download=True, transform=transform)
                trn_load = torch.utils.data.DataLoader(trnset, batch_size=50000, shuffle=False, num_workers=0)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=0)
                self.channels = 3;
                self.width = 32;
                self.height = 32;
                self.n_cls = 100;

            if self.dataset != 'emnist':
                trn_itr = trn_load.__iter__();
                tst_itr = tst_load.__iter__()
                # labels are of shape (n_data,)
                trn_x, trn_y = trn_itr.__next__()
                tst_x, tst_y = tst_itr.__next__()

                trn_x = trn_x.numpy();
                trn_y = trn_y.numpy().reshape(-1, 1)
                tst_x = tst_x.numpy();
                tst_y = tst_y.numpy().reshape(-1, 1)

            if self.dataset == 'emnist':
                emnist = io.loadmat(self.data_path + "/Raw/matlab/emnist-letters.mat")
                # load training dataset
                x_train = emnist["dataset"][0][0][0][0][0][0]
                x_train = x_train.astype(np.float32)

                # load training labels
                y_train = emnist["dataset"][0][0][0][0][0][1] - 1  # make first class 0

                # take first 10 classes of letters
                trn_idx = np.where(y_train < 10)[0]

                y_train = y_train[trn_idx]
                x_train = x_train[trn_idx]

                mean_x = np.mean(x_train)
                std_x = np.std(x_train)

                # load test dataset
                x_test = emnist["dataset"][0][0][1][0][0][0]
                x_test = x_test.astype(np.float32)

                # load test labels
                y_test = emnist["dataset"][0][0][1][0][0][1] - 1  # make first class 0

                tst_idx = np.where(y_test < 10)[0]

                y_test = y_test[tst_idx]
                x_test = x_test[tst_idx]

                x_train = x_train.reshape((-1, 1, 28, 28))
                x_test = x_test.reshape((-1, 1, 28, 28))

                # normalise train and test features

                trn_x = (x_train - mean_x) / std_x
                trn_y = y_train

                tst_x = (x_test - mean_x) / std_x
                tst_y = y_test

                self.channels = 1;
                self.width = 28;
                self.height = 28;
                self.n_cls = 10;

            # Shuffle Data
            rand_perm = np.random.permutation(len(trn_y))
            trn_x = trn_x[rand_perm]
            trn_y = trn_y[rand_perm]

            self.trn_x = trn_x
            self.trn_y = trn_y
            self.tst_x = tst_x
            self.tst_y = tst_y

            ###
            n_data_per_clnt = int((len(trn_y)) / self.n_client)
            if self.unbalanced_sgm != 0:
                # Draw from lognormal distribution
                clnt_data_list = (
                    np.random.lognormal(mean=np.log(n_data_per_clnt), sigma=self.unbalanced_sgm, size=self.n_client))
                clnt_data_list = (clnt_data_list / np.sum(clnt_data_list) * len(trn_y)).astype(int)
                diff = np.sum(clnt_data_list) - len(trn_y)

                # Add/Subtract the excess number starting from first client
                if diff != 0:
                    for clnt_i in range(self.n_client):
                        if clnt_data_list[clnt_i] > diff:
                            clnt_data_list[clnt_i] -= diff
                            break
            else:
                clnt_data_list = (np.ones(self.n_client) * n_data_per_clnt).astype(int)
            ###     

            if self.rule == 'Dirichlet':
                cls_priors = np.random.dirichlet(alpha=[self.rule_arg] * self.n_cls, size=self.n_client)
                prior_cumsum = np.cumsum(cls_priors, axis=1)
                idx_list = [np.where(trn_y == i)[0] for i in range(self.n_cls)]
                cls_amount = [len(idx_list[i]) for i in range(self.n_cls)]

                clnt_x = [np.zeros((clnt_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32)
                          for clnt__ in range(self.n_client)]
                clnt_y = [np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client)]

                while (np.sum(clnt_data_list) != 0):
                    curr_clnt = np.random.randint(self.n_client)
                    # If current node is full resample a client
                    print('Remaining Data: %d' % np.sum(clnt_data_list))
                    if clnt_data_list[curr_clnt] <= 0:
                        continue
                    clnt_data_list[curr_clnt] -= 1
                    curr_prior = prior_cumsum[curr_clnt]
                    while True:
                        cls_label = np.argmax(np.random.uniform() <= curr_prior)
                        # Redraw class label if trn_y is out of that class
                        if cls_amount[cls_label] <= 0:
                            continue
                        cls_amount[cls_label] -= 1
                        clnt_x[curr_clnt][clnt_data_list[curr_clnt]] = trn_x[idx_list[cls_label][cls_amount[cls_label]]]
                        clnt_y[curr_clnt][clnt_data_list[curr_clnt]] = trn_y[idx_list[cls_label][cls_amount[cls_label]]]

                        break

                clnt_x = np.asarray(clnt_x)
                clnt_y = np.asarray(clnt_y)

                cls_means = np.zeros((self.n_client, self.n_cls))
                for clnt in range(self.n_client):
                    for cls in range(self.n_cls):
                        cls_means[clnt, cls] = np.mean(clnt_y[clnt] == cls)
                prior_real_diff = np.abs(cls_means - cls_priors)
                print('--- Max deviation from prior: %.4f' % np.max(prior_real_diff))
                print('--- Min deviation from prior: %.4f' % np.min(prior_real_diff))

            elif self.rule == 'iid' and self.dataset == 'CIFAR100' and self.unbalanced_sgm == 0:
                assert len(trn_y) // 100 % self.n_client == 0
                # Only have the number clients if it divides 500
                # Perfect IID partitions for cifar100 instead of shuffling
                idx = np.argsort(trn_y[:, 0])
                n_data_per_clnt = len(trn_y) // self.n_client
                # clnt_x dtype needs to be float32, the same as weights
                clnt_x = np.zeros((self.n_client, n_data_per_clnt, 3, 32, 32), dtype=np.float32)
                clnt_y = np.zeros((self.n_client, n_data_per_clnt, 1), dtype=np.float32)
                trn_x = trn_x[idx]  # 50000*3*32*32
                trn_y = trn_y[idx]
                n_cls_sample_per_device = n_data_per_clnt // 100
                for i in range(self.n_client):  # devices
                    for j in range(100):  # class
                        clnt_x[i, n_cls_sample_per_device * j:n_cls_sample_per_device * (j + 1), :, :, :] = trn_x[
                                                                                                            500 * j + n_cls_sample_per_device * i:500 * j + n_cls_sample_per_device * (
                                                                                                                        i + 1),
                                                                                                            :, :, :]
                        clnt_y[i, n_cls_sample_per_device * j:n_cls_sample_per_device * (j + 1), :] = trn_y[
                                                                                                      500 * j + n_cls_sample_per_device * i:500 * j + n_cls_sample_per_device * (
                                                                                                                  i + 1),
                                                                                                      :]


            elif self.rule == 'iid':

                clnt_x = [np.zeros((clnt_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32)
                          for clnt__ in range(self.n_client)]
                clnt_y = [np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client)]

                clnt_data_list_cum_sum = np.concatenate(([0], np.cumsum(clnt_data_list)))
                for clnt_idx_ in range(self.n_client):
                    clnt_x[clnt_idx_] = trn_x[clnt_data_list_cum_sum[clnt_idx_]:clnt_data_list_cum_sum[clnt_idx_ + 1]]
                    clnt_y[clnt_idx_] = trn_y[clnt_data_list_cum_sum[clnt_idx_]:clnt_data_list_cum_sum[clnt_idx_ + 1]]

                clnt_x = np.asarray(clnt_x)
                clnt_y = np.asarray(clnt_y)

            self.clnt_x = clnt_x;
            self.clnt_y = clnt_y

            self.tst_x = tst_x;
            self.tst_y = tst_y

            # Save data
            os.mkdir('%s/%s' % (self.data_path, self.name))

            np.save('%s/%s/clnt_x.npy' % (self.data_path, self.name), clnt_x)
            np.save('%s/%s/clnt_y.npy' % (self.data_path, self.name), clnt_y)

            np.save('%s/%s/tst_x.npy' % (self.data_path, self.name), tst_x)
            np.save('%s/%s/tst_y.npy' % (self.data_path, self.name), tst_y)

        else:
            print("Data is already downloaded in the folder.")
            self.clnt_x = np.load('%s/%s/clnt_x.npy' % (self.data_path, self.name), allow_pickle=True)
            self.clnt_y = np.load('%s/%s/clnt_y.npy' % (self.data_path, self.name), allow_pickle=True)
            self.n_client = len(self.clnt_x)

            self.tst_x = np.load('%s/%s/tst_x.npy' % (self.data_path, self.name), allow_pickle=True)
            self.tst_y = np.load('%s/%s/tst_y.npy' % (self.data_path, self.name), allow_pickle=True)

            if self.dataset == 'mnist':
                self.channels = 1;
                self.width = 28;
                self.height = 28;
                self.n_cls = 10;
            if self.dataset == 'CIFAR10':
                self.channels = 3;
                self.width = 32;
                self.height = 32;
                self.n_cls = 10;
            if self.dataset == 'CIFAR100':
                self.channels = 3;
                self.width = 32;
                self.height = 32;
                self.n_cls = 100;
            if self.dataset == 'fashion_mnist':
                self.channels = 1;
                self.width = 28;
                self.height = 28;
                self.n_cls = 10;
            if self.dataset == 'emnist':
                self.channels = 1;
                self.width = 28;
                self.height = 28;
                self.n_cls = 10;

        print('Class frequencies:')
        count = 0
        for clnt in range(self.n_client):
            print("Client %3d: " % clnt +
                  ', '.join(["%.3f" % np.mean(self.clnt_y[clnt] == cls) for cls in range(self.n_cls)]) +
                  ', Amount:%d' % self.clnt_y[clnt].shape[0])
            count += self.clnt_y[clnt].shape[0]

        print('Total Amount:%d' % count)
        print('--------')

        print("      Test: " +
              ', '.join(["%.3f" % np.mean(self.tst_y == cls) for cls in range(self.n_cls)]) +
              ', Amount:%d' % self.tst_y.shape[0])


def generate_syn_logistic(dimension, n_clnt, n_cls, avg_data=4, alpha=1.0, beta=0.0, theta=0.0, iid_sol=False,
                          iid_dat=False):
    # alpha is for minimizer of each client
    # beta  is for distirbution of points
    # theta is for number of data points

    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j + 1), -1.2)
    cov_x = np.diag(diagonal)

    samples_per_user = (np.random.lognormal(mean=np.log(avg_data + 1e-3), sigma=theta, size=n_clnt)).astype(int)
    print('samples per user')
    print(samples_per_user)
    print('sum %d' % np.sum(samples_per_user))

    num_samples = np.sum(samples_per_user)

    data_x = list(range(n_clnt))
    data_y = list(range(n_clnt))

    mean_W = np.random.normal(0, alpha, n_clnt)
    B = np.random.normal(0, beta, n_clnt)

    mean_x = np.zeros((n_clnt, dimension))

    if not iid_dat:  # If IID then make all 0s.
        for i in range(n_clnt):
            mean_x[i] = np.random.normal(B[i], 1, dimension)

    sol_W = np.random.normal(mean_W[0], 1, (dimension, n_cls))
    sol_B = np.random.normal(mean_W[0], 1, (1, n_cls))

    if iid_sol:  # Then make vectors come from 0 mean distribution
        sol_W = np.random.normal(0, 1, (dimension, n_cls))
        sol_B = np.random.normal(0, 1, (1, n_cls))

    for i in range(n_clnt):
        if not iid_sol:
            sol_W = np.random.normal(mean_W[i], 1, (dimension, n_cls))
            sol_B = np.random.normal(mean_W[i], 1, (1, n_cls))

        data_x[i] = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        data_y[i] = np.argmax((np.matmul(data_x[i], sol_W) + sol_B), axis=1).reshape(-1, 1)

    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y)
    return data_x, data_y


class DatasetSynthetic:
    def __init__(self, alpha, beta, theta, iid_sol, iid_data, n_dim, n_clnt, n_cls, avg_data, name_prefix):
        self.dataset = 'synt'
        self.name = name_prefix + '_'
        self.name += '%d_%d_%d_%d_%f_%f_%f_%s_%s' % (n_dim, n_clnt, n_cls, avg_data,
                                                     alpha, beta, theta, iid_sol, iid_data)

        data_path = 'Data'
        if (not os.path.exists('%s/%s/' % (data_path, self.name))):
            # Generate data
            print('Sythetize')
            data_x, data_y = generate_syn_logistic(dimension=n_dim, n_clnt=n_clnt, n_cls=n_cls, avg_data=avg_data,
                                                   alpha=alpha, beta=beta, theta=theta,
                                                   iid_sol=iid_sol, iid_dat=iid_data)
            os.mkdir('%s/%s/' % (data_path, self.name))
            np.save('%s/%s/data_x.npy' % (data_path, self.name), data_x)
            np.save('%s/%s/data_y.npy' % (data_path, self.name), data_y)
        else:
            # Load data
            print('Load')
            data_x = np.load('%s/%s/data_x.npy' % (data_path, self.name), allow_pickle=True)
            data_y = np.load('%s/%s/data_y.npy' % (data_path, self.name), allow_pickle=True)

        for clnt in range(n_clnt):
            print(', '.join(['%.4f' % np.mean(data_y[clnt] == t) for t in range(n_cls)]))

        self.clnt_x = data_x
        self.clnt_y = data_y

        self.tst_x = np.concatenate(self.clnt_x, axis=0)
        self.tst_y = np.concatenate(self.clnt_y, axis=0)
        self.n_client = len(data_x)
        print(self.clnt_x.shape)


# Added by Jonathan
def create_dir_if_not_exists(dir_path):
    if not path.exists(path.dirname(dir_path)):
        try:
            os.makedirs(path.dirname(dir_path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def prepare_datasets(X, Y):
    # Flatten
    X = np.concatenate(X, axis=0)
    if Y is not None:
        Y = np.concatenate(Y, axis=0)
    # Trim
    ###X = remove_coordinates(X)
    return X, Y


def load_fd_data_set(base_path, file_name, n_clnt_dataset):
    X = [[] for _ in range(n_clnt_dataset)]
    Y = [[] for _ in range(n_clnt_dataset)]
    raw_data_path = base_path + '/' + file_name

    with open(raw_data_path, 'rb') as f:
        print("Loading partial x's")
        for nid in range(n_clnt_dataset):
            partial_x = pickle.load(f)
            np.save('%s/partial_x_%d.npy' % (base_path, nid), partial_x)
            print(nid)

        print("Loading partial y's")
        for nid in range(n_clnt_dataset):
            partial_y = pickle.load(f)
            np.save('%s/partial_y_%d.npy' % (base_path, nid), partial_y)
            print(nid)

    print("Merging partials")
    for nid in range(n_clnt_dataset):
        X[nid] = np.load('%s/partial_x_%d.npy' % (base_path, nid))
        Y[nid] = np.load('%s/partial_y_%d.npy' % (base_path, nid))

    print("Done with Load")
    return X, Y


def load_and_prepare_dataset(base_path, file_name, n_clnt_dataset):
    X, Y = load_fd_data_set(base_path, file_name, n_clnt_dataset)
    np.random.seed(1337)

    X, Y = prepare_datasets(X, Y)

    # Shuffle training sets
    X, Y = shuffle(X, Y, random_state=1337)

    # Create 50/50 ratio for positives and negatives
    false_mask = np.where(Y == 0)[0]
    true_mask = Y == 1
    num_true = np.sum(Y)
    keep_mask = np.random.choice(false_mask, num_true, replace=False)
    mask_array = np.zeros(len(Y), dtype=bool)
    mask_array[keep_mask] = True

    # Select the datapoints chosen for correct ratio, and trim to make computation time feasible
    X = X[true_mask | mask_array]  # [:2000000]
    Y = Y[true_mask | mask_array]  # [:2000000]

    return X, Y

class DatasetFD:
    def __init__(self, n_clnt_federated, n_clnt_dataset, name_prefix):
        """
        Parameters:
             n_clnt_federated (int): The number of clients that the federated learning architecture should use.
             n_clnt_dataset (int): The number of clients that was used in the reliable broadcast deployment
             where the logs were collected from.
             name_prefix (str): Prefix used for this particular test set.
        """

        self.dataset = 'fd'
        self.name = name_prefix + '_'
        self.name += '%d' % n_clnt_federated
        n_cls = 2
        self.n_clnt_dataset = n_clnt_dataset

        data_path = 'Data'
        if not os.path.exists('%s/%s/' % (data_path, self.name)):
            create_dir_if_not_exists('%s/%s/' % (data_path, self.name))
            # Generate data
            print("Data does not exist in correct format. Creating it...")
            X_train, Y_train = load_and_prepare_dataset('%s/Raw/%s' % (data_path, self.dataset), 'train.pkl', 7)
            print("Loaded Training Data")

            excess_data = len(X_train) % n_clnt_federated
            X_train = X_train[:-excess_data]
            Y_train = Y_train[:-excess_data]

            print("Train Length:", len(Y_train))
            print("True ratio: ", np.sum(Y_train) / len(Y_train))

            clnt_x = np.array(np.split(X_train, n_clnt_federated))
            clnt_y = np.array(np.split(Y_train, n_clnt_federated))

            np.save('%s/%s/clnt_x.npy' % (data_path, self.name), clnt_x)
            np.save('%s/%s/clnt_y.npy' % (data_path, self.name), clnt_y)

            X_test, Y_test = load_and_prepare_dataset('%s/Raw/%s' % (data_path, self.dataset), 'test.pkl', 7)

            np.save('%s/%s/tst_x.npy' % (data_path, self.name), X_test)
            np.save('%s/%s/tst_y.npy' % (data_path, self.name), Y_test)

            tst_x = X_test
            tst_y = Y_test

            print("Len Y_test:", len(Y_test))
            print("True ratio: ", np.sum(Y_test) / len(Y_test))

        else:
            # Load data
            print('Load')
            clnt_x = np.load('%s/%s/clnt_x.npy' % (data_path, self.name), allow_pickle=True)
            clnt_y = np.load('%s/%s/clnt_y.npy' % (data_path, self.name), allow_pickle=True)

            tst_x = np.load('%s/%s/tst_x.npy' % (data_path, self.name), allow_pickle=True)
            tst_y = np.load('%s/%s/tst_y.npy' % (data_path, self.name), allow_pickle=True)

        for clnt in range(n_clnt_federated):
            print(', '.join(['%.4f' % np.mean(clnt_y[clnt] == t) for t in range(n_cls)]))

        self.clnt_x = clnt_x
        self.clnt_y = clnt_y

        self.tst_x = tst_x
        self.tst_y = tst_y
        self.n_client = len(clnt_x)
        print(self.clnt_x.shape)


def load_fd_data_set2(base_path, n_clnt_dataset, deployment_id=-1, train=False):
    if deployment_id != -1:
        base_path += f'/deployment_{deployment_id}/'
    path_X = base_path + ('train_red_X.npz' if train else 'test_red_X.npz')
    path_Y = base_path + ('train_red_Y.npz' if train else 'test_red_Y.npz')
    print(f"Loading from {path_X} and {path_Y}")
    with open(path_X, 'rb') as f:
        X = pickle.load(f)
    with open(path_Y, 'rb') as f:
        Y = pickle.load(f)
    return X, Y

class MI(IntEnum): # Meta indices
    TIME_START = 0,
    TIME_END = 1,
    NID = 2,
    IS_MUTE = 3,
    SEQ = 4,
    RTC = 5,
    TSH = 6

def filter_datapoints(vecs, mets):
    for nidx, nid_mets in enumerate(mets):
        indices_to_remove = []
        for i, met in enumerate(nid_mets):
            if met[MI.TSH] >= 180000:
                indices_to_remove.append(i)

        vecs[nidx] = np.delete(vecs[nidx], indices_to_remove, axis=0)
        mets[nidx] = np.delete(mets[nidx], indices_to_remove, axis=0)

def load_fd_data_set3(base_path, n_clnt_dataset, deployment_id=-1, train=False):
    vecs = [[] for _ in range(n_clnt_dataset)]
    mets = [[] for _ in range(n_clnt_dataset)]
    path = base_path + ('/train/' if train else '/test/')
    if deployment_id != -1:
        path += f'deployment_{deployment_id}/'

    path += 'red/'

    print(f"Loading from {path}")
    filenames = next(walk(path), (None, None, []))[2]  # [] if no file
    for f_name in sorted(filenames):
        nid = int(f_name.split('_')[1].split('.')[0])
        print('Opening file ', f_name)
        loaded_data = np.load(path + f_name)
        vecs[nid] = loaded_data['vec']
        mets[nid] = loaded_data['met']

    filter_datapoints(vecs, mets)

    X = [[] for _ in range(n_clnt_dataset)]
    Y = [[] for _ in range(n_clnt_dataset)]
    for nid in range(n_clnt_dataset):
        for i in range(len(vecs[nid])):
            X[nid].append(vecs[nid][i])
            Y[nid].append(mets[nid][i][MI.IS_MUTE])
        X[nid] = np.array(X[nid])
        Y[nid] = np.array(Y[nid])

    return X, Y


def load_and_prepare_dataset2(base_path, deployment_id, n_clnt_dataset, train=False):
    X, Y = load_fd_data_set3(base_path, n_clnt_dataset=n_clnt_dataset, deployment_id=deployment_id, train=train)
    np.random.seed(1337)

    X, Y = prepare_datasets(X, Y)

    # Shuffle training sets
    X, Y = shuffle(X, Y, random_state=1337)

    # Create 50/50 ratio for positives and negatives
    false_mask = np.where(Y == 0)[0]
    true_mask = Y == 1
    num_true = np.sum(Y)
    keep_mask = np.random.choice(false_mask, int(num_true), replace=False)
    mask_array = np.zeros(len(Y), dtype=bool)
    mask_array[keep_mask] = True

    # Select the datapoints chosen for correct ratio, and trim to make computation time feasible
    X = X[true_mask | mask_array]  # [:2000000]
    Y = Y[true_mask | mask_array]  # [:2000000]

    return X, Y

class DatasetFD2:
    def __init__(self, n_clnt_federated, deployment_id, name_prefix):
        """
        Parameters:
             n_clnt_federated (int): The number of clients that the federated learning architecture should use.
             deployment_id (int): The id of the deployment that the dataset corresponds to.
             name_prefix (str): Prefix used for this particular test set.
        """
        deployment_constants = {
            0: {'N_CLIENTS': 7, 'NUM_BYZ': 2},
            1: {'N_CLIENTS': 6, 'NUM_BYZ': 1},
            2: {'N_CLIENTS': 6, 'NUM_BYZ': 1},
            3: {'N_CLIENTS': 7, 'NUM_BYZ': 2},
            4: {'N_CLIENTS': 7, 'NUM_BYZ': 2},
            5: {'N_CLIENTS': 7, 'NUM_BYZ': 2},
            6: {'N_CLIENTS': 6, 'NUM_BYZ': 1},
            7: {'N_CLIENTS': 6, 'NUM_BYZ': 1}
        }
        if deployment_id == -1:
            dep_ids = [k for k in deployment_constants.keys()]
        else:
            dep_ids = [deployment_id]
        self.dataset = 'fd'
        self.name = f'{name_prefix}{deployment_id}_'
        self.name += '%d' % n_clnt_federated
        n_cls = 2
        # self.n_clnt_dataset = n_clnt_dataset

        data_path = 'Data'
        if not os.path.exists('%s/%s/' % (data_path, self.name)):
            create_dir_if_not_exists('%s/%s/' % (data_path, self.name))
            clnt_x = [[] for _ in dep_ids]
            clnt_y = [[] for _ in dep_ids]
            print("Data does not exist in correct format. Creating it...")
            clnt_shortest = -1
            for dep_id in dep_ids:
                dep_consts = deployment_constants[dep_id]
                n_clnt_dataset = dep_consts['N_CLIENTS']

                # Generate data
                clnt_x[dep_id], clnt_y[dep_id] = load_and_prepare_dataset2('%s/Raw/%s' % (data_path, self.dataset),
                                                             deployment_id=dep_id,
                                                             n_clnt_dataset=n_clnt_dataset,
                                                             train=True)
                print(f"Loaded Training Data for deployment {dep_id}")

                #excess_data = len(X_train_dep) % n_clnt_federated
                #if excess_data != 0:
                #    X_train = X_train_dep[:-excess_data]
                #    Y_train = Y_train_dep[:-excess_data]

                len_y = len(clnt_y[dep_id])
                print("Train Length:", len_y)
                print("True ratio: ", np.sum(clnt_y[dep_id]) / len_y)
                if len(clnt_y[dep_id]) < clnt_shortest or clnt_shortest == -1:
                    clnt_shortest = len_y

            clnt_x = [np.array(sublist[:clnt_shortest]) for sublist in clnt_x]
            clnt_y = [np.array(sublist[:clnt_shortest]) for sublist in clnt_y]

            #clnt_x = np.array(np.split(X_train, n_clnt_federated))
            #clnt_y = np.array(np.split(Y_train, n_clnt_federated))
            clnt_x = np.array(clnt_x)
            clnt_y = np.array(clnt_y)
            np.save('%s/%s/clnt_x.npy' % (data_path, self.name), clnt_x)
            np.save('%s/%s/clnt_y.npy' % (data_path, self.name), clnt_y)

            tst_x = []
            tst_y = []
            for dep_id in dep_ids:
                dep_consts = deployment_constants[dep_id]
                n_clnt_dataset = dep_consts['N_CLIENTS']
                test_X, test_Y = load_and_prepare_dataset2('%s/Raw/%s' % (data_path, self.dataset),
                                                             deployment_id=dep_id,
                                                             n_clnt_dataset=n_clnt_dataset,
                                                             train=False)
                print(f"Loaded Testing Data for deployment {dep_id}")
                tst_x.append(test_X)
                tst_y.append(test_Y)
            tst_x = np.concatenate(tst_x)
            tst_y = np.concatenate(tst_y)
            len_y = len(tst_y)
            print("Test Length:", len_y)
            print("True ratio: ", np.sum(tst_y) / len_y)
            np.save('%s/%s/tst_x.npy' % (data_path, self.name), tst_x)
            np.save('%s/%s/tst_y.npy' % (data_path, self.name), tst_y)


        else:
            # Load data
            print('Load')
            clnt_x = np.load('%s/%s/clnt_x.npy' % (data_path, self.name), allow_pickle=True)
            clnt_y = np.load('%s/%s/clnt_y.npy' % (data_path, self.name), allow_pickle=True)

            tst_x = np.load('%s/%s/tst_x.npy' % (data_path, self.name), allow_pickle=True)
            tst_y = np.load('%s/%s/tst_y.npy' % (data_path, self.name), allow_pickle=True)

        for clnt in range(n_clnt_federated):
            print(', '.join(['%.4f' % np.mean(clnt_y[clnt] == t) for t in range(n_cls)]))

        self.clnt_x = clnt_x
        self.clnt_y = clnt_y

        self.tst_x = tst_x
        self.tst_y = tst_y
        self.n_client = len(clnt_x)
        print(self.clnt_x.shape)


# Original prepration is from LEAF paper...
# This loads Shakespeare dataset only.
# data_path/train and data_path/test are assumed to be processed
# To make the dataset smaller,
# We take 2000 datapoints for each client in the train_set

class ShakespeareObjectCrop:
    def __init__(self, data_path, dataset_prefix, crop_amount=2000, tst_ratio=5, rand_seed=0):
        self.dataset = 'shakespeare'
        self.name = dataset_prefix
        users, groups, train_data, test_data = read_data(data_path + 'train/', data_path + 'test/')

        # train_data is a dictionary whose keys are users list elements
        # the value of each key is another dictionary.
        # This dictionary consists of key value pairs as 
        # (x, features - list of input 80 lenght long words) and (y, target - list one letter)
        # test_data has the same strucute.
        # Ignore groups information, combine test cases for different clients into one test data
        # Change structure to DatasetObject structure

        self.users = users

        self.n_client = len(users)
        self.user_idx = np.asarray(list(range(self.n_client)))
        self.clnt_x = list(range(self.n_client))
        self.clnt_y = list(range(self.n_client))

        tst_data_count = 0

        for clnt in range(self.n_client):
            np.random.seed(rand_seed + clnt)
            start = np.random.randint(len(train_data[users[clnt]]['x']) - crop_amount)
            self.clnt_x[clnt] = np.asarray(train_data[users[clnt]]['x'])[start:start + crop_amount]
            self.clnt_y[clnt] = np.asarray(train_data[users[clnt]]['y'])[start:start + crop_amount]

        tst_data_count = (crop_amount // tst_ratio) * self.n_client
        self.tst_x = list(range(tst_data_count))
        self.tst_y = list(range(tst_data_count))

        tst_data_count = 0
        for clnt in range(self.n_client):
            curr_amount = (crop_amount // tst_ratio)
            np.random.seed(rand_seed + clnt)
            start = np.random.randint(len(test_data[users[clnt]]['x']) - curr_amount)
            self.tst_x[tst_data_count: tst_data_count + curr_amount] = np.asarray(test_data[users[clnt]]['x'])[
                                                                       start:start + curr_amount]
            self.tst_y[tst_data_count: tst_data_count + curr_amount] = np.asarray(test_data[users[clnt]]['y'])[
                                                                       start:start + curr_amount]

            tst_data_count += curr_amount

        self.clnt_x = np.asarray(self.clnt_x)
        self.clnt_y = np.asarray(self.clnt_y)

        self.tst_x = np.asarray(self.tst_x)
        self.tst_y = np.asarray(self.tst_y)

        # Convert characters to numbers

        self.clnt_x_char = np.copy(self.clnt_x)
        self.clnt_y_char = np.copy(self.clnt_y)

        self.tst_x_char = np.copy(self.tst_x)
        self.tst_y_char = np.copy(self.tst_y)

        self.clnt_x = list(range(len(self.clnt_x_char)))
        self.clnt_y = list(range(len(self.clnt_x_char)))

        for clnt in range(len(self.clnt_x_char)):
            clnt_list_x = list(range(len(self.clnt_x_char[clnt])))
            clnt_list_y = list(range(len(self.clnt_x_char[clnt])))

            for idx in range(len(self.clnt_x_char[clnt])):
                clnt_list_x[idx] = np.asarray(word_to_indices(self.clnt_x_char[clnt][idx]))
                clnt_list_y[idx] = np.argmax(np.asarray(letter_to_vec(self.clnt_y_char[clnt][idx]))).reshape(-1)

            self.clnt_x[clnt] = np.asarray(clnt_list_x)
            self.clnt_y[clnt] = np.asarray(clnt_list_y)

        self.clnt_x = np.asarray(self.clnt_x)
        self.clnt_y = np.asarray(self.clnt_y)

        self.tst_x = list(range(len(self.tst_x_char)))
        self.tst_y = list(range(len(self.tst_x_char)))

        for idx in range(len(self.tst_x_char)):
            self.tst_x[idx] = np.asarray(word_to_indices(self.tst_x_char[idx]))
            self.tst_y[idx] = np.argmax(np.asarray(letter_to_vec(self.tst_y_char[idx]))).reshape(-1)

        self.tst_x = np.asarray(self.tst_x)
        self.tst_y = np.asarray(self.tst_y)


class ShakespeareObjectCrop_noniid:
    def __init__(self, data_path, dataset_prefix, n_client=100, crop_amount=2000, tst_ratio=5, rand_seed=0):
        self.dataset = 'shakespeare'
        self.name = dataset_prefix
        users, groups, train_data, test_data = read_data(data_path + 'train/', data_path + 'test/')

        # train_data is a dictionary whose keys are users list elements
        # the value of each key is another dictionary.
        # This dictionary consists of key value pairs as 
        # (x, features - list of input 80 lenght long words) and (y, target - list one letter)
        # test_data has the same strucute.        
        # Change structure to DatasetObject structure

        self.users = users

        tst_data_count_per_clnt = (crop_amount // tst_ratio)
        # Group clients that have at least crop_amount datapoints
        arr = []
        for clnt in range(len(users)):
            if (len(np.asarray(train_data[users[clnt]]['y'])) > crop_amount
                    and len(np.asarray(test_data[users[clnt]]['y'])) > tst_data_count_per_clnt):
                arr.append(clnt)

        # choose n_client clients randomly
        self.n_client = n_client
        np.random.seed(rand_seed)
        np.random.shuffle(arr)
        self.user_idx = arr[:self.n_client]

        self.clnt_x = list(range(self.n_client))
        self.clnt_y = list(range(self.n_client))

        tst_data_count = 0

        for clnt, idx in enumerate(self.user_idx):
            np.random.seed(rand_seed + clnt)
            start = np.random.randint(len(train_data[users[idx]]['x']) - crop_amount)
            self.clnt_x[clnt] = np.asarray(train_data[users[idx]]['x'])[start:start + crop_amount]
            self.clnt_y[clnt] = np.asarray(train_data[users[idx]]['y'])[start:start + crop_amount]

        tst_data_count = (crop_amount // tst_ratio) * self.n_client
        self.tst_x = list(range(tst_data_count))
        self.tst_y = list(range(tst_data_count))

        tst_data_count = 0

        for clnt, idx in enumerate(self.user_idx):
            curr_amount = (crop_amount // tst_ratio)
            np.random.seed(rand_seed + clnt)
            start = np.random.randint(len(test_data[users[idx]]['x']) - curr_amount)
            self.tst_x[tst_data_count: tst_data_count + curr_amount] = np.asarray(test_data[users[idx]]['x'])[
                                                                       start:start + curr_amount]
            self.tst_y[tst_data_count: tst_data_count + curr_amount] = np.asarray(test_data[users[idx]]['y'])[
                                                                       start:start + curr_amount]
            tst_data_count += curr_amount

        self.clnt_x = np.asarray(self.clnt_x)
        self.clnt_y = np.asarray(self.clnt_y)

        self.tst_x = np.asarray(self.tst_x)
        self.tst_y = np.asarray(self.tst_y)

        # Convert characters to numbers

        self.clnt_x_char = np.copy(self.clnt_x)
        self.clnt_y_char = np.copy(self.clnt_y)

        self.tst_x_char = np.copy(self.tst_x)
        self.tst_y_char = np.copy(self.tst_y)

        self.clnt_x = list(range(len(self.clnt_x_char)))
        self.clnt_y = list(range(len(self.clnt_x_char)))

        for clnt in range(len(self.clnt_x_char)):
            clnt_list_x = list(range(len(self.clnt_x_char[clnt])))
            clnt_list_y = list(range(len(self.clnt_x_char[clnt])))

            for idx in range(len(self.clnt_x_char[clnt])):
                clnt_list_x[idx] = np.asarray(word_to_indices(self.clnt_x_char[clnt][idx]))
                clnt_list_y[idx] = np.argmax(np.asarray(letter_to_vec(self.clnt_y_char[clnt][idx]))).reshape(-1)

            self.clnt_x[clnt] = np.asarray(clnt_list_x)
            self.clnt_y[clnt] = np.asarray(clnt_list_y)

        self.clnt_x = np.asarray(self.clnt_x)
        self.clnt_y = np.asarray(self.clnt_y)

        self.tst_x = list(range(len(self.tst_x_char)))
        self.tst_y = list(range(len(self.tst_x_char)))

        for idx in range(len(self.tst_x_char)):
            self.tst_x[idx] = np.asarray(word_to_indices(self.tst_x_char[idx]))
            self.tst_y[idx] = np.argmax(np.asarray(letter_to_vec(self.tst_y_char[idx]))).reshape(-1)

        self.tst_x = np.asarray(self.tst_x)
        self.tst_y = np.asarray(self.tst_y)


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_x, data_y=True, train=False, dataset_name=''):
        self.name = dataset_name
        if self.name == 'mnist' or self.name == 'synt' or self.name == 'emnist':
            self.X_data = torch.tensor(data_x).float()
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = torch.tensor(data_y).float()

        elif self.name == 'CIFAR10' or self.name == 'CIFAR100':
            self.train = train
            self.transform = transforms.Compose([transforms.ToTensor()])

            self.X_data = data_x
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = data_y.astype('float32')

        elif self.name == 'shakespeare':

            self.X_data = data_x
            self.y_data = data_y

            self.X_data = torch.tensor(self.X_data).long()
            if not isinstance(data_y, bool):
                self.y_data = torch.tensor(self.y_data).float()

        elif self.name == 'fd':
            self.X_data = data_x
            self.y_data = data_y

            self.X_data = torch.tensor(self.X_data).float()
            if not isinstance(data_y, bool):
                self.y_data = torch.tensor(self.y_data).float()

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        if self.name == 'mnist' or self.name == 'synt' or self.name == 'emnist':
            X = self.X_data[idx, :]
            if isinstance(self.y_data, bool):
                return X
            else:
                y = self.y_data[idx]
                return X, y

        elif self.name == 'CIFAR10' or self.name == 'CIFAR100':
            img = self.X_data[idx]
            if self.train:
                img = np.flip(img, axis=2).copy() if (np.random.rand() > .5) else img  # Horizontal flip
                if (np.random.rand() > .5):
                    # Random cropping
                    pad = 4
                    extended_img = np.zeros((3, 32 + pad * 2, 32 + pad * 2)).astype(np.float32)
                    extended_img[:, pad:-pad, pad:-pad] = img
                    dim_1, dim_2 = np.random.randint(pad * 2 + 1, size=2)
                    img = extended_img[:, dim_1:dim_1 + 32, dim_2:dim_2 + 32]
            img = np.moveaxis(img, 0, -1)
            img = self.transform(img)
            if isinstance(self.y_data, bool):
                return img
            else:
                y = self.y_data[idx]
                return img, y

        elif self.name == 'shakespeare':
            x = self.X_data[idx]
            y = self.y_data[idx]
            return x, y

        elif self.name == 'fd':
            x = self.X_data[idx]
            y = self.y_data[idx]
            return x, y

