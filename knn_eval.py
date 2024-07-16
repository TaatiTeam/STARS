import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import numpy as np

from feeder.ntu_feeder import Feeder_single
from net.transformer import Transformer

from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=list, default=[1, 2, 5, 10], nargs='+', help='which Top K accuracy will be shown')
    parser.add_argument('--data-path', type=str, default='./data/NTU60_XSub.npz')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num-workers', type=int, default=32)
    return parser.parse_args()

def test_extract_hidden(model, data_train, data_eval):
    model.eval()
    print("Extracting training features")
    label_train_list = []
    hidden_array_train_list = []
    for ith, (ith_data, label) in enumerate(tqdm(data_train)):
            input_tensor = ith_data.cuda()

            en_hi = model(input_tensor, return_feature=True)
            en_hi = en_hi.squeeze()
            #print("encoder size",en_hi.size())

            label_train_list.append(label)
            hidden_array_train_list.append(en_hi[:, :].detach().cpu().numpy())
    label_train = np.hstack(label_train_list)
    hidden_array_train = np.vstack(hidden_array_train_list)

    print("Extracting validation features")
    label_eval_list = []
    hidden_array_eval_list = []
    for ith, (ith_data,  label) in enumerate(tqdm(data_eval)):

        input_tensor = ith_data.cuda()

        en_hi = model(input_tensor, return_feature=True)
        en_hi = en_hi.squeeze()

        label_eval_list.append(label)
        hidden_array_eval_list.append(en_hi[:, :].detach().cpu().numpy())
    label_eval = np.hstack(label_eval_list)
    hidden_array_eval = np.vstack(hidden_array_eval_list)

    return hidden_array_train, hidden_array_eval, label_train, label_eval

def knn(data_train, data_test, label_train, label_test, nn=9):
    label_train = np.asarray(label_train)
    label_test = np.asarray(label_test)
    print("Number of KNN Neighbours = ", nn)
    print("training feature and labels", data_train.shape, len(label_train))
    print("test feature and labels", data_test.shape, len(label_test))

    Xtr_Norm = preprocessing.normalize(data_train)
    Xte_Norm = preprocessing.normalize(data_test)

    knn = KNeighborsClassifier(n_neighbors=nn,
                               metric='cosine')  # , metric='cosine'#'mahalanobis', metric_params={'V': np.cov(data_train)})
    knn.fit(Xtr_Norm, label_train)
    pred = knn.predict(Xte_Norm)
    # if nn == 10:
    #     np.save('all_pred.npz', pred)
    #     np.save('all_labels.npz', label_test)
    acc = accuracy_score(pred, label_test)

    return acc


def clustering_knn_acc(model, train_loader, eval_loader, knn_neighbours):
    hi_train, hi_eval, label_train, label_eval = test_extract_hidden(model, train_loader, eval_loader)
    
    knn_results = {}
    for k in knn_neighbours:
        knn_results[k] = knn(hi_train, hi_eval, label_train, label_eval, nn=k)

    return knn_results

def main():
    args = parse_args()
    print(f"preparing dataset...")
    train_dataset = Feeder_single(data_path=args.data_path,
                                   p_interval=[0.95],
                                   split='train',
                                   window_size=120,
                                   shear_amplitude=-1,
                                   aug_method='')
    val_dataset = Feeder_single(data_path=args.data_path,
                                  p_interval=[0.95],
                                  split='test',
                                  window_size=120,
                                  shear_amplitude=-1,
                                  aug_method='')
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                             shuffle=False,  drop_last=False, num_workers=args.num_workers,
                             pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                             shuffle=False,  drop_last=False, num_workers=args.num_workers,
                             pin_memory=True)
    
    model = Transformer(dim_in=3, dim_feat=256, depth=8, num_heads=8, mlp_ratio=4,
                        num_frames=120, num_joints=25, patch_size=1, t_patch_size=4, qkv_bias=True,
                        qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                        cls_token=False)
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    print(f"Load pre-trained checkpoint from: {args.checkpoint}")
    if 'model' in checkpoint: # MAMP
        checkpoint = checkpoint['model']
        checkpoint_model = {}
        for key in checkpoint:
            if not (key.startswith('decoder') or key.startswith('mask')):
                checkpoint_model[key] = checkpoint[key]
    else: # STARS
        checkpoint_model = {}
        for key in checkpoint:
            if key.startswith('encoder_student') and '.head' not in key:
                checkpoint_model[key.replace('encoder_student.', '')] = checkpoint[key]

    msg = model.load_state_dict(checkpoint_model, strict=False)
    assert set(msg.missing_keys) == {'head.weight', 'head.bias'}, set(msg.missing_keys)
    
    for p in model.parameters():
        p.requires_grad = False

    model.cuda()
    model.eval()

    results = clustering_knn_acc(model, train_loader, val_loader, args.k)
    for k in results:
        print(f"k={k}: {results[k]}")



if __name__ == '__main__':
    main()