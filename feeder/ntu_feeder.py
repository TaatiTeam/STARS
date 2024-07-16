import numpy as np
import pickle, torch
try:
    from . import tools
except ImportError:
    import tools


class Feeder_single(torch.utils.data.Dataset):
    """ Feeder for single input (120 frames)"""

    def __init__(self, data_path, p_interval=1, split='train', window_size=-1,
                 shear_amplitude=0.5, mmap=True, aug_method=''):
        self.data_path = data_path
        self.split = split
        self.p_interval = p_interval
        self.window_size = window_size
        self.aug_method = aug_method

        self.shear_amplitude = shear_amplitude
       
        self.load_data(mmap)

    def load_data(self, mmap):
        # data: N C V T M
        if mmap:
            npz_data = np.load(self.data_path, mmap_mode='r')
        else:
            npz_data = np.load(self.data_path)

        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')

        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)

        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        data_numpy = self._aug(data_numpy)
        
        return data_numpy, label
    
    def _aug(self, data_numpy):
        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        if '1' in self.aug_method:
            data_numpy = tools.random_spatial_flip(data_numpy)
        if '2' in self.aug_method:
            data_numpy = tools.random_rotate(data_numpy)
        if '3' in self.aug_method:
            data_numpy = tools.gaus_noise(data_numpy)
        if '4' in self.aug_method:
            data_numpy = tools.gaus_filter(data_numpy)
        if '5' in self.aug_method:
            data_numpy = tools.axis_mask(data_numpy)
        if '6' in self.aug_method:
            data_numpy = tools.random_time_flip(data_numpy)
        if '7' in self.aug_method:
            data_numpy = tools.warp_time(data_numpy)
        
        return data_numpy

class Feeder_dual(torch.utils.data.Dataset):
    """ Feeder for dual inputs (120 frames)"""

    def __init__(self, data_path, p_interval=1, split='train', window_size=-1,
                 shear_amplitude=0.5, mmap=True, aug_method='125'):
        self.data_path = data_path
        self.split = split
        self.p_interval = p_interval
        self.window_size = window_size
        self.aug_method = aug_method

        self.shear_amplitude = shear_amplitude
       
        self.load_data(mmap)

    def load_data(self, mmap):
        # data: N C V T M
        if mmap:
            npz_data = np.load(self.data_path, mmap_mode='r')
        else:
            npz_data = np.load(self.data_path)

        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')

        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = np.array(self.data[index])
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)

        all_data = []
        data_numpy_crop = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        # local views                
        for _ in range(2):
            data = self._aug(data_numpy_crop)
            all_data.append(data)

        return all_data, index

    def _aug(self, data_numpy):
        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        if '1' in self.aug_method:
            data_numpy = tools.random_spatial_flip(data_numpy)
        if '2' in self.aug_method:
            data_numpy = tools.random_rotate(data_numpy)
        if '3' in self.aug_method:
            data_numpy = tools.gaus_noise(data_numpy)
        if '4' in self.aug_method:
            data_numpy = tools.gaus_filter(data_numpy)
        if '5' in self.aug_method:
            data_numpy = tools.axis_mask(data_numpy)
        if '6' in self.aug_method:
            data_numpy = tools.random_time_flip(data_numpy)
        if '7' in self.aug_method:
            data_numpy = tools.warp_time(data_numpy)
        
        return data_numpy
    

class Feeder_multi(torch.utils.data.Dataset):
    """ Feeder for multi inputs (120 frames)"""

    def __init__(self, data_path, p_interval=1, split='train', window_size=-1,
                 shear_amplitude=0.5, mmap=True, aug_method='125', n_globals=2, n_locals=4):
        self.data_path = data_path
        self.split = split
        self.p_interval = p_interval
        self.window_size = window_size
        self.aug_method = aug_method
        self.n_globals = n_globals
        self.n_locals = n_locals

        self.shear_amplitude = shear_amplitude
       
        self.load_data(mmap)

    def load_data(self, mmap):
        # data: N C V T M
        if mmap:
            npz_data = np.load(self.data_path, mmap_mode='r')
        else:
            npz_data = np.load(self.data_path)

        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')

        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = np.array(self.data[index])
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)

        all_data = []
        # reshape Tx(MVC) to CTVM
        data_numpy_v1_crop = tools.valid_crop_resize(data_numpy, valid_frame_num, [1], self.window_size)
        # global views
        for _ in range(self.n_globals):
            all_data.append(self._aug(data_numpy_v1_crop))

        data_numpy_v2_crop = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        # local views                
        for _ in range(self.n_locals):
            data = self._aug(data_numpy_v2_crop)
            all_data.append(data)

        return all_data, index

    def _aug(self, data_numpy):
        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        if '1' in self.aug_method:
            data_numpy = tools.random_spatial_flip(data_numpy)
        if '2' in self.aug_method:
            data_numpy = tools.random_rotate(data_numpy)
        if '3' in self.aug_method:
            data_numpy = tools.gaus_noise(data_numpy)
        if '4' in self.aug_method:
            data_numpy = tools.gaus_filter(data_numpy)
        if '5' in self.aug_method:
            data_numpy = tools.axis_mask(data_numpy)
        if '6' in self.aug_method:
            data_numpy = tools.random_time_flip(data_numpy)
        if '7' in self.aug_method:
            data_numpy = tools.warp_time(data_numpy)
        
        return data_numpy
    

def _test():
    from visualize import visualize_sequence, NTU_PAIRS
    dataset = Feeder_multi(data_path='../data/NTU60_XSub.npz',
                            p_interval=[1.0],
                            window_size=120,
                            split='train',
                            shear_amplitude=-1,
                            aug_method='',
                            mmap=True)
    print(len(dataset))
    seq_number = 30
    sequence = dataset[seq_number][0][0]
    sequence = torch.from_numpy(sequence)
    sequence = sequence.permute(3, 1, 2, 0)  # (C, T, V, M) -> (M, T, V, C)
    sequence = sequence[0]  # First person
    sequence = sequence.numpy()
    visualize_sequence(sequence, NTU_PAIRS, name=f'sequence{seq_number}')



if __name__ == '__main__':
    _test()