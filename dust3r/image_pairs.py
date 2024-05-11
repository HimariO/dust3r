# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilities needed to load image pairs
# --------------------------------------------------------
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd


SUPERPOINT_CKPT = "/home/ron/Documents/ImageMatching/pretrain_models/superpoint_v1.pth"
LIGTHGLUE_CKPT = "/home/ron/Documents/ImageMatching/pretrain_models/superpoint_lightglue.pth"
DINO_CKPT = 'facebook/dinov2-base'

# SUPERPOINT_CKPT = "//kaggle/input/imc-models/pretrain_models/superpoint_v1.pth"
# LIGTHGLUE_CKPT = "/kaggle/input/imc-models/pretrain_models/superpoint_lightglue.pth"
# DINO_CKPT = '/kaggle/input/dinov2/pytorch/base/1/'


@torch.no_grad
def make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True, filelist=None):
    pairs = []
    if scene_graph == 'complete':  # complete graph
        for i in range(len(imgs)):
            for j in range(i):
                pairs.append((imgs[i], imgs[j]))
    elif scene_graph.startswith('swin'):
        winsize = int(scene_graph.split('-')[1]) if '-' in scene_graph else 3
        pairsid = set()
        for i in range(len(imgs)):
            for j in range(1, winsize+1):
                idx = (i + j) % len(imgs)  # explicit loop closure
                pairsid.add((i, idx) if i < idx else (idx, i))
        for i, j in pairsid:
            pairs.append((imgs[i], imgs[j]))
    elif scene_graph.startswith('oneref'):
        refid = int(scene_graph.split('-')[1]) if '-' in scene_graph else 0
        for j in range(len(imgs)):
            if j != refid:
                pairs.append((imgs[refid], imgs[j]))
    elif scene_graph.startswith('lightglue'):
        topk = int(scene_graph.split('-')[1]) if '-' in scene_graph else 0
        extractor = SuperPoint(max_num_keypoints=2048, checkpoint=SUPERPOINT_CKPT).eval().cuda()  # load the extractor
        matcher = LightGlue(features='superpoint', checkpoint=SUPERPOINT_CKPT).eval().cuda()  # load the matcher
        
        trange = tqdm(range(len(imgs)), desc="make lightglue pairs")
        for i in trange:
            feats0 = extractor.extract(load_image(filelist[i]).cuda())
            match_list = []
            for j in range(i):
                feats1 = extractor.extract(load_image(filelist[j]).cuda())
                matches01 = matcher({'image0': feats0, 'image1': feats1})
                matches01 = rbd(matches01)
                match_pts = matches01['matches'].shape[0]
                if match_pts > 0:
                    match_list.append((j, match_pts))
            
            match_list = sorted(match_list, key=lambda x: -x[1])
            for j, _ in match_list[:topk]:
                pairs.append((imgs[i], imgs[j]))
    elif scene_graph.startswith('dino'):
        from transformers import AutoImageProcessor, AutoModel
        processor = AutoImageProcessor.from_pretrained(DINO_CKPT)
        model = AutoModel.from_pretrained(DINO_CKPT).eval().to('cuda')

        topk = int(scene_graph.split('-')[1]) if '-' in scene_graph else 0
        trange = tqdm(range(len(imgs)), desc="make dino pairs")
        embeds = []
        
        for i in trange:
            img_i = Image.open(filelist[i])
            inputs = processor(images=img_i, return_tensors="pt").to('cuda')
            outputs = model(**inputs)
            embed = torch.nn.functional.normalize(outputs.last_hidden_state.mean(dim=1))
            embeds.append(torch.squeeze(embed, dim=0))
            
            match_list = []
            for j in range(i):
                score = torch.dot(embeds[j], embeds[i]).cpu()
                if score > .3:
                    match_list.append((score, j))
            
            if match_list:
                match_list = sorted(match_list)
                step_size = max(1, len(match_list) // topk)
                for sc, j in match_list[step_size - 1::step_size]:
                    pairs.append((imgs[i], imgs[j]))
                print(len(match_list[step_size - 1::step_size]), len(match_list))
        
    if symmetrize:
        pairs += [(img2, img1) for img1, img2 in pairs]

    # now, remove edges
    if isinstance(prefilter, str) and prefilter.startswith('seq'):
        pairs = filter_pairs_seq(pairs, int(prefilter[3:]))

    if isinstance(prefilter, str) and prefilter.startswith('cyc'):
        pairs = filter_pairs_seq(pairs, int(prefilter[3:]), cyclic=True)

    return pairs


def sel(x, kept):
    if isinstance(x, dict):
        return {k: sel(v, kept) for k, v in x.items()}
    if isinstance(x, (torch.Tensor, np.ndarray)):
        return x[kept]
    if isinstance(x, (tuple, list)):
        return type(x)([x[k] for k in kept])


def _filter_edges_seq(edges, seq_dis_thr, cyclic=False):
    # number of images
    n = max(max(e) for e in edges)+1

    kept = []
    for e, (i, j) in enumerate(edges):
        dis = abs(i-j)
        if cyclic:
            dis = min(dis, abs(i+n-j), abs(i-n-j))
        if dis <= seq_dis_thr:
            kept.append(e)
    return kept


def filter_pairs_seq(pairs, seq_dis_thr, cyclic=False):
    edges = [(img1['idx'], img2['idx']) for img1, img2 in pairs]
    kept = _filter_edges_seq(edges, seq_dis_thr, cyclic=cyclic)
    return [pairs[i] for i in kept]


def filter_edges_seq(view1, view2, pred1, pred2, seq_dis_thr, cyclic=False):
    edges = [(int(i), int(j)) for i, j in zip(view1['idx'], view2['idx'])]
    kept = _filter_edges_seq(edges, seq_dis_thr, cyclic=cyclic)
    print(f'>> Filtering edges more than {seq_dis_thr} frames apart: kept {len(kept)}/{len(edges)} edges')
    return sel(view1, kept), sel(view2, kept), sel(pred1, kept), sel(pred2, kept)
