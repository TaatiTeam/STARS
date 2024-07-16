import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from feeder.ntu_feeder import Feeder_single
from net.transformer import Transformer

color_list = [
    '#e6194B', # Red
    '#3cb44b', # Green
    '#ffe119', # Yellow
    '#4363d8', # Blue
    '#f58231', # Orange
    '#42d4f4', # Cyan
    '#f032e6', # Magneta
    '#fabed4', # Pink
    '#469990', # Teal
    '#dcbeff', # Lavender
    '#9A6324', # Brown
    '#800000', # Maroon
    '#aaffc3', # Mint
    '#000075', # Navy
    '#a9a9a9', # Grey
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actions', type=list, default=[1, 2, 5, 10, 15, 18, 20, 22, 25, 28, 30, 40, 50, 55, 57], nargs='+')
    parser.add_argument('--data-path', type=str, default='./data/NTU60_XSub.npz')
    parser.add_argument('--num-samples', type=int, default=100)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num-workers', type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    assert len(args.actions) <= 15, "Actions cannot be more than 15 for visualization purpose"

    print(f"preparing dataset...")
    val_dataset = Feeder_single(data_path=args.data_path,
                                  p_interval=[0.95],
                                  split='test',
                                  window_size=120,
                                  shear_amplitude=-1,
                                  aug_method='')
    val_loader = DataLoader(dataset=val_dataset, batch_size=1,
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

    action_count = {action: 0 for action in args.actions}
    features, labels = [], []
    for sequence, label in tqdm(val_loader):
        sequence = sequence.cuda()
        label = label.item()
        if len(labels) == len(args.actions) * args.num_samples:
            break
        if label in args.actions:
            if action_count[label] < args.num_samples:
                action_count[label] += 1
            else:
                continue
        else:
            continue
        
        feature = model(sequence, return_feature=True)
        features.append(feature)
        labels.append(label)
    
    features = torch.cat(features, dim=0).cpu().numpy()
    labels = np.array(labels)

    tsne = TSNE(n_components=2, random_state=42, learning_rate='auto', init='pca')
    tsne_result = tsne.fit_transform(features)

    color_dict = {action: color_list[idx] for idx, action in enumerate(args.actions)}
    colors = [color_dict[label] for label in labels]

    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=colors, s=5)
    plt.xticks([])
    plt.yticks([])
    plt.title("STARS")
    plt.savefig('tsne.png', dpi=1200)
        



if __name__ == '__main__':
    main()