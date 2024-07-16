import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from math import sin, cos

transform_order = {
    'ntu': [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 16, 17, 18, 19, 12, 13, 14, 15, 20, 23, 24, 21, 22]
}

trunk_ori_index = [4, 3, 21, 2, 1]
left_hand_ori_index = [9, 10, 11, 12, 24, 25]
right_hand_ori_index = [5, 6, 7, 8, 22, 23]
left_leg_ori_index = [17, 18, 19, 20]
right_leg_ori_index = [13, 14, 15, 16]

trunk = [i - 1 for i in trunk_ori_index]
left_hand = [i - 1 for i in left_hand_ori_index]
right_hand = [i - 1 for i in right_hand_ori_index]
left_leg = [i - 1 for i in left_leg_ori_index]
right_leg = [i - 1 for i in right_leg_ori_index]
body_parts = [trunk, left_hand, right_hand, left_leg, right_leg]


def shear(data_numpy, r=0.5):

    center_trans = data_numpy[:, :, 20, :].copy()
    data_numpy[:, :, 20, :] = 0.0

    s1_list = [random.uniform(-r, r), random.uniform(-r,
                                                     r), random.uniform(-r, r)]
    s2_list = [random.uniform(-r, r), random.uniform(-r,
                                                     r), random.uniform(-r, r)]

    R = np.array([[1,          s1_list[0], s2_list[0]],
                  [s1_list[1], 1,          s2_list[1]],
                  [s1_list[2], s2_list[2], 1]])

    R = R.transpose()
    data_numpy = np.dot(data_numpy.transpose([1, 2, 3, 0]), R)
    data_numpy = data_numpy.transpose(3, 0, 1, 2)

    data_numpy[:, :, 20, :] = center_trans

    return data_numpy


def temperal_crop(data_numpy, temperal_padding_ratio=6):
    C, T, V, M = data_numpy.shape
    padding_len = T // temperal_padding_ratio
    frame_start = np.random.randint(0, padding_len * 2 + 1)
    data_numpy = np.concatenate((data_numpy[:, :padding_len][:, ::-1],
                                 data_numpy,
                                 data_numpy[:, -padding_len:][:, ::-1]),
                                axis=1)
    data_numpy = data_numpy[:, frame_start:frame_start + T]
    return data_numpy


def rescale(data_numpy, scale_rate=0.0):
    Bone = [(21, 21), (2, 21), (3, 21), (9, 21), (5, 21), (1, 2), (4, 3), (6, 5), (7, 6), (8, 7),
            (10, 9), (11, 10), (12, 11), (13, 1), (14,
                                                   13), (15, 14), (16, 15), (17, 1),
            (18, 17), (19, 18), (20, 19), (23, 8), (22, 23), (25, 12), (24, 25)]

    bone = np.zeros_like(data_numpy)

    for v1, v2 in Bone:
        bone[:, :, v1 - 1, :] = data_numpy[:, :,
                                           v1 - 1, :] - data_numpy[:, :, v2 - 1, :]

    for v1, v2 in Bone:
        data_numpy[:, :, v1 - 1, :] = data_numpy[:, :, v2 - 1, :] + \
            bone[:, :, v1 - 1, :] * scale_rate * random.random()

    # C, T, V, M = data_numpy.shape
    # mean = np.mean(data_numpy, axis=(3), keepdims=True)
    # std = np.std(data_numpy, axis=(2), keepdims=True)
    # data_numpy = (data_numpy - mean) / (std + 1e-3) * (scale_rate * random.random() + 1e-3) + (scale_rate * random.random())
    return data_numpy


def random_spatial_flip(seq, p=0.5):
    if random.random() < p:
        # Do the left-right transform C,T,V,M
        index = transform_order['ntu']
        trans_seq = np.copy(seq)
        trans_seq[0] *= -1
        trans_seq = trans_seq[:, :, index, :]
        return trans_seq
    else:
        return seq


def random_time_flip(seq, p=0.5):
    T = seq.shape[1]
    if random.random() < p:
        time_range_order = [i for i in range(T)]
        time_range_reverse = list(reversed(time_range_order))
        return seq[:, time_range_reverse, :, :]
    else:
        return seq


def ske_swap(data1, data2):
    '''
    swap a batch skeleton
    T   64 --> 32 --> 16    # 8n
    S   25 --> 25 --> 25 (5 parts)
    '''
    spa_l, spa_u, tem_l, tem_u = 2, 3, 15, 35

    T = data1.shape[1]

    # ------ Spatial ------ #
    Cs = random.randint(spa_l, spa_u)
    # sample the parts index
    parts_idx = random.sample(body_parts, Cs)
    # generate spa_idx
    spa_idx = []
    for part_idx in parts_idx:
        spa_idx += part_idx
    spa_idx.sort()

    # ------ Temporal ------ #
    Ct = random.randint(tem_l, tem_u)
    tem_idx = random.randint(0, T - Ct)
    rt = Ct

    p = random.random()

    if p > 0.5:
        # begin swap
        data1[:, tem_idx: tem_idx + rt, spa_idx,
              :] = data2[:, tem_idx: tem_idx + rt, spa_idx, :]
    elif p <= 0.5 and p > 0.25:
        data2 = data2[:, ::(T // rt), spa_idx, :]
        t = data2.shape[1]
        rt = min(rt, t)
        data1[:, tem_idx: tem_idx + rt, spa_idx, :] = data2[:, :rt, :, :]
    elif p <= 0.25:
        lamb = random.random()
        data1 = data1 * (1 - lamb) + data2 * lamb

    return data1


def random_rotate(seq):
    def rotate(seq, axis, angle):
        # x
        if axis == 0:
            R = np.array([[1, 0, 0],
                          [0, cos(angle), sin(angle)],
                          [0, -sin(angle), cos(angle)]])
        # y
        if axis == 1:
            R = np.array([[cos(angle), 0, -sin(angle)],
                          [0, 1, 0],
                          [sin(angle), 0, cos(angle)]])

        # z
        if axis == 2:
            R = np.array([[cos(angle), sin(angle), 0],
                          [-sin(angle), cos(angle), 0],
                          [0, 0, 1]])
        R = R.T
        temp = np.matmul(seq, R)
        return temp

    new_seq = seq.copy()
    # C, T, V, M -> T, V, M, C
    new_seq = np.transpose(new_seq, (1, 2, 3, 0))
    total_axis = [0, 1, 2]
    main_axis = random.randint(0, 2)
    for axis in total_axis:
        if axis == main_axis:
            rotate_angle = random.uniform(0, 30)
            rotate_angle = math.radians(rotate_angle)
            new_seq = rotate(new_seq, axis, rotate_angle)
        else:
            rotate_angle = random.uniform(0, 1)
            rotate_angle = math.radians(rotate_angle)
            new_seq = rotate(new_seq, axis, rotate_angle)

    new_seq = np.transpose(new_seq, (3, 0, 1, 2))

    return new_seq


def gaus_noise(data_numpy, mean=0, std=0.01, p=0.5):
    if random.random() < p:
        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape
        noise = np.random.normal(mean, std, size=(C, T, V, M))
        return temp + noise
    else:
        return data_numpy


def gaus_filter(data_numpy):
    g = GaussianBlurConv(3)
    return g(data_numpy)


def dropout(data_numpy, p=0.5):
    # if random.random() < p:
    mask = np.random.uniform(size=data_numpy.shape) > p
    return mask * data_numpy
    # else:
    #     return data_numpy


class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3, kernel=15, sigma=[0.1, 2]):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        self.kernel = kernel
        self.min_max_sigma = sigma
        radius = int(kernel / 2)
        self.kernel_index = np.arange(-radius, radius + 1)

    def __call__(self, x):
        sigma = random.uniform(self.min_max_sigma[0], self.min_max_sigma[1])
        blur_flter = np.exp(-np.power(self.kernel_index,
                            2.0) / (2.0 * np.power(sigma, 2.0)))
        kernel = torch.from_numpy(blur_flter).unsqueeze(0).unsqueeze(0)
        # kernel =  kernel.float()
        kernel = kernel.double()
        kernel = kernel.repeat(self.channels, 1, 1, 1)  # (3,1,1,5)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

        prob = np.random.random_sample()
        x = torch.from_numpy(x)
        if prob < 0.5:
            x = x.permute(3, 0, 2, 1)  # M,C,V,T
            x = F.conv2d(x, self.weight, padding=(
                0, int((self.kernel - 1) / 2)),   groups=self.channels)
            x = x.permute(1, -1, -2, 0)  # C,T,V,M

        return x.numpy()


class Zero_out_axis(object):
    def __init__(self, axis=None):
        self.first_axis = axis

    def __call__(self, data_numpy):
        if self.first_axis != None:
            axis_next = self.first_axis
        else:
            axis_next = random.randint(0, 2)

        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape
        x_new = np.zeros((T, V, M))
        temp[axis_next] = x_new
        return temp


def axis_mask(data_numpy, p=0.5):
    am = Zero_out_axis()
    if random.random() < p:
        return am(data_numpy)
    else:
        return data_numpy


def valid_crop_resize(data_numpy, valid_frame_num, p_interval, window):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num
    valid_size = end - begin

    # crop
    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1-p) * valid_size/2)
        data = data_numpy[:, begin+bias:end-bias, :, :]  # center_crop
        cropped_length = data.shape[1]
    else:
        p = np.random.rand(1)*(p_interval[1]-p_interval[0])+p_interval[0]
        # constraint cropped_length lower bound as 64
        cropped_length = np.minimum(np.maximum(
            int(np.floor(valid_size*p)), 64), valid_size)
        bias = np.random.randint(0, valid_size-cropped_length+1)
        data = data_numpy[:, begin+bias:begin+bias+cropped_length, :, :]
        if data.shape[1] == 0:
            print(cropped_length, bias, valid_size)

    # resize
    data = torch.tensor(data, dtype=torch.float)
    data = data.permute(0, 2, 3, 1).contiguous().view(
        C * V * M, cropped_length)
    data = data[None, None, :, :]
    # could perform both up sample and down sample
    data = F.interpolate(data, size=(C * V * M, window),
                         mode='bilinear', align_corners=False).squeeze()
    data = data.contiguous().view(C, V, M, window).permute(
        0, 3, 1, 2).contiguous().numpy()

    return data


def temporal_cropresize(input_data, num_of_frames, l_ratio, output_size):

    C, T, V, M = input_data.shape

    # Temporal crop
    min_crop_length = 64

    scale = np.random.rand(1)*(l_ratio[1]-l_ratio[0])+l_ratio[0]
    temporal_crop_length = np.minimum(np.maximum(
        int(np.floor(num_of_frames*scale)), min_crop_length), num_of_frames)

    start = np.random.randint(0, num_of_frames-temporal_crop_length+1)
    temporal_context = input_data[:, start:start+temporal_crop_length, :, :]

    # interpolate
    temporal_context = torch.tensor(temporal_context, dtype=torch.float)
    temporal_context = temporal_context.permute(
        0, 2, 3, 1).contiguous().view(C * V * M, temporal_crop_length)
    temporal_context = temporal_context[None, :, :, None]
    temporal_context = F.interpolate(temporal_context, size=(
        output_size, 1), mode='bilinear', align_corners=False)
    temporal_context = temporal_context.squeeze(dim=3).squeeze(dim=0)
    temporal_context = temporal_context.contiguous().view(
        C, V, M, output_size).permute(0, 3, 1, 2).contiguous().numpy()

    return temporal_context


def warp_time(data_numpy):
    C, T, V, M = data_numpy.shape

    cycle = np.random.randint(low=1, high=17)
    freq = np.linspace(0, cycle * np.pi, T)
    freq_warped = np.abs(np.sin(freq)) if np.random.randint(
        0, 2) == 0 else np.abs(np.cos(freq))
    normalized_freq_warped = freq_warped / np.sum(freq_warped)
    discretized_freq_warped = np.round(
        normalized_freq_warped * T).astype(np.int32)

    frequency_length = int(np.sum(discretized_freq_warped))
    if frequency_length > T:
        remained = frequency_length - T
        positive_indices = np.where(discretized_freq_warped > 0)[0]
        selected_indices = np.random.choice(
            positive_indices, size=remained, replace=False)
        discretized_freq_warped[selected_indices] -= 1
    elif frequency_length < T:
        remained = T - frequency_length
        zero_indices = np.where(discretized_freq_warped == 0)[0]
        selected_indices = np.random.choice(
            zero_indices, size=remained, replace=False)
        discretized_freq_warped[selected_indices] += 1

    assert np.sum(discretized_freq_warped) == T
    indices = np.arange(T)
    indices = np.repeat(indices, discretized_freq_warped)

    return np.ascontiguousarray(data_numpy[:, indices])


def crop_subsequence(input_data, num_of_frames, l_ratio, output_size):

    C, T, V, M = input_data.shape

    if l_ratio[0] == 0.5:
        # if training , sample a random crop

        min_crop_length = 64
        scale = np.random.rand(1)*(l_ratio[1]-l_ratio[0])+l_ratio[0]
        temporal_crop_length = np.minimum(np.maximum(
            int(np.floor(num_of_frames*scale)), min_crop_length), num_of_frames)

        start = np.random.randint(0, num_of_frames-temporal_crop_length+1)
        temporal_crop = input_data[:, start:start+temporal_crop_length, :, :]

        temporal_crop = torch.tensor(temporal_crop, dtype=torch.float)
        temporal_crop = temporal_crop.permute(
            0, 2, 3, 1).contiguous().view(C * V * M, temporal_crop_length)
        temporal_crop = temporal_crop[None, :, :, None]
        temporal_crop = F.interpolate(temporal_crop, size=(
            output_size, 1), mode='bilinear', align_corners=False)
        temporal_crop = temporal_crop.squeeze(dim=3).squeeze(dim=0)
        temporal_crop = temporal_crop.contiguous().view(
            C, V, M, output_size).permute(0, 3, 1, 2).contiguous().numpy()

        return temporal_crop

    else:
        # if testing , sample a center crop

        start = int((1-l_ratio[0]) * num_of_frames/2)
        data = input_data[:, start:num_of_frames-start, :, :]
        temporal_crop_length = data.shape[1]

        temporal_crop = torch.tensor(data, dtype=torch.float)
        temporal_crop = temporal_crop.permute(
            0, 2, 3, 1).contiguous().view(C * V * M, temporal_crop_length)
        temporal_crop = temporal_crop[None, :, :, None]
        temporal_crop = F.interpolate(temporal_crop, size=(
            output_size, 1), mode='bilinear', align_corners=False)
        temporal_crop = temporal_crop.squeeze(dim=3).squeeze(dim=0)
        temporal_crop = temporal_crop.contiguous().view(
            C, V, M, output_size).permute(0, 3, 1, 2).contiguous().numpy()

        return temporal_crop


if __name__ == '__main__':
    data_seq = np.ones((3, 50, 25, 2))
    data_seq = axis_mask(data_seq)
    print(data_seq.shape)
