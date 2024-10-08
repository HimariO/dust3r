# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilities needed to load image pairs
# --------------------------------------------------------

import math
from collections import defaultdict, deque
from itertools import product
from heapq import heapify, heappop, heappush
from typing import *

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


class LightglueMatcher:

    def __init__(self) -> None:
        self.extractor = SuperPoint(max_num_keypoints=2048, checkpoint=SUPERPOINT_CKPT).eval().cuda()  # load the extractor
        self.matcher = LightGlue(features='superpoint', checkpoint=LIGTHGLUE_CKPT).eval().cuda()  # load the matcher
    
    def load_img(self, img_dict) -> torch.Tensor:
        img_path = img_dict['filepath']
        return load_image(img_path).cuda()

    def rotate_img_90(self, img: Union[Image.Image, torch.Tensor], k=1):
        if k < 1:
            return img
        return torch.rot90(img, k=k, dims=(1, 2))

    def extract_feat(self, img: torch.Tensor) -> Any:
        return self.extractor.extract(img)
    
    def num_match_points(self, feats0, feats1) -> int:
        matches01 = self.matcher({'image0': feats0, 'image1': feats1})
        matches01 = rbd(matches01)
        match_pts = matches01['matches'].shape[0]
        return match_pts
    
    def get_pair_matches(self, imgs: List[Any], pairs_idx: List[Tuple[int]]=None):
        n = len(imgs)
        if pairs_idx is None:
            pairs_idx = [
                (i, j) 
                for i in range(n) 
                for j in range(i)
            ]
        
        pair_matches = defaultdict(lambda: 0)
        feats = defaultdict(list)
        for i in range(n):
            for r in range(4):
                feat = self.extract_feat(
                    self.rotate_img_90(self.load_img(imgs[i]), k=r)
                )
                feats[r].append(feat)
        
        rotate_states = [0] * n
        for pair in tqdm(pairs_idx, desc=f"{self.__class__.__name__} pairing"):
            i, j = pair[:2]
            # for r1, r2 in product(range(4), range(4)):
            for r2 in range(4):
                feats0, feats1 = feats[0][i], feats[r2][j]
                match_pts = self.num_match_points(feats0, feats1)
                if match_pts > pair_matches[i, j]:
                    rotate_states[j] = (rotate_states[i] + r2) % 4
                pair_matches[i, j] = max(pair_matches[i, j], match_pts)
        for i in range(n):
            imgs[i]['img'] = torch.rot90(
                imgs[i]['img'], 
                k=rotate_states[i], 
                dims=(2, 3)
            )
            if rotate_states[i] % 2:
                imgs[i]['true_shape'] = imgs[i]['true_shape'][..., ::-1].copy()
        return pair_matches



def MST_pairing(edges: DefaultDict[Tuple[int], float], n: int, k: int):
    _graph = defaultdict(dict)
    for (a, b), w in edges.items():
        _graph[a][b] = w
        _graph[b][a] = w
    
    pairs_idx = []
    for (a, b), w in list(edges.items()):
        edges[b, a] = w
    
    def mark_components(start: int):
        nonlocal edges, node_groups, num_groups
        que = deque([start])
        while que:
            node = que.popleft()
            if node in node_groups:
                continue
            
            node_groups[node] = num_groups
            for i in range(n):
                if (node, i) in edges and i not in node_groups:
                    que.append(i)
    
    def prim(start: int):
        nonlocal pairs_idx
        visit = [False] * n
        degrees = [0] * n
        que = [
            (degrees[i], edges[start, i], start, i) 
            for i in range(n)
            if i != start and (start, i) in edges
        ]
        heapify(que)
        
        print(f" [prim]")
        while que:
            deg, d, a, b = heappop(que)
            if deg != max(degrees[a], degrees[b]):
                heappush(que, (max(degrees[a], degrees[b]), d, a, b))
                continue
            
            if visit[b] or (not d < math.inf):
                continue
            print(f"{deg}/{degrees[b]}", f"{d:.2f}", f"{a} <--> {b}")
            visit[a] = True
            visit[b] = True
            pairs_idx.append((a, b, 1 - d))
            del edges[a, b]
            del edges[b, a]
            degrees[a] += 1
            degrees[b] += 1
            
            for i in range(n):
                if i != b and (b, i) in edges and not visit[i]:
                    heappush(que, (max(degrees[b], degrees[i]), edges[b, i], b, i))
        return
    
    num_groups = 0
    node_groups = {}
    group2node = defaultdict(list)

    for _ in range(k):
        for node in range(n):
            if node not in node_groups:
                mark_components(node)
                num_groups += 1
        for node, group in node_groups.items():
            group2node[group].append(node)
        
        for group, nodes in group2node.items():
            prim(nodes[0])
        
        # _debug = Counter([a for a, b, c in pairs_idx] + [b for a, b, c in pairs_idx])
        num_groups = 0
        node_groups = {}
        group2node = defaultdict(list)
    return pairs_idx


@torch.no_grad
def make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True, filelist=None):
    n = len(imgs)
    pairs = []
    
    if scene_graph == 'complete':  # complete graph
        for i in range(n):
            for j in range(i):
                pairs.append((imgs[i], imgs[j]))
    elif scene_graph.startswith('swin'):
        winsize = int(scene_graph.split('-')[1]) if '-' in scene_graph else 3
        pairsid = set()
        for i in range(n):
            for j in range(1, winsize+1):
                idx = (i + j) % n  # explicit loop closure
                pairsid.add((i, idx) if i < idx else (idx, i))
        for i, j in pairsid:
            pairs.append((imgs[i], imgs[j]))
    elif scene_graph.startswith('oneref'):
        refid = int(scene_graph.split('-')[1]) if '-' in scene_graph else 0
        for j in range(n):
            if j != refid:
                pairs.append((imgs[refid], imgs[j]))
    elif scene_graph.startswith('lightglue'):
        topk = int(scene_graph.split('-')[1]) if '-' in scene_graph else 0
        assert topk > 0
        matcher = LightglueMatcher()
        match_points_dict = matcher.get_pair_matches(imgs)
        
        mst = True
        trange = tqdm(range(n), desc="make lightglue pairs")
        if mst:
            norm_matches = defaultdict(lambda: math.inf)
            max_pts = max(match_points_dict.values())
            for i in trange:
                match_list = []
                for j in range(i):
                    norm_matches[i, j] = 1 - match_points_dict[i, j] / max_pts
            pairs_idx = MST_pairing(norm_matches, n, topk)
            for i, j, _ in pairs_idx:
                pairs.append((imgs[i], imgs[j]))
        else:
            for i in trange:
                match_list = []
                for j in range(i):
                    match_pts = match_points_dict[i, j]
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
        trange = tqdm(range(n), desc="make dino pairs")
        mst = True
        
        if mst:
            embeds = {}
            inv_cos_sim = defaultdict(lambda: math.inf)
            
            for i in trange:
                img_i = Image.open(filelist[i])
                inputs = processor(images=img_i, return_tensors="pt").to('cuda')
                outputs = model(**inputs)
                embed = torch.nn.functional.normalize(outputs.last_hidden_state.mean(dim=1))
                embeds[i] = torch.squeeze(embed, dim=0)
                
                for j in range(i):
                    score = 1 - float(torch.dot(embeds[j], embeds[i]).cpu())
                    if score > .3:
                        inv_cos_sim[j, i] = score
            
            pairs_idx = MST_pairing(inv_cos_sim, n, topk)
            for i, j, _ in pairs_idx:
                pairs.append((imgs[i], imgs[j]))
        else:
            embeds = []
            for i in trange:
                img_i = Image.open(filelist[i])
                inputs = processor(images=img_i, return_tensors="pt").to('cuda')
                outputs = model(**inputs)
                embed = torch.nn.functional.normalize(outputs.last_hidden_state.mean(dim=1))
                embeds.append(torch.squeeze(embed, dim=0))
                
                match_list = []
                for j in range(i):
                    score = float(torch.dot(embeds[j], embeds[i]).cpu())
                    if score > .3:
                        match_list.append((score, j))
                
                if match_list:
                    match_list = sorted(match_list, reverse=True)
                    # NOTE: skip high simliarity images when we have more images than we need.
                    num_distinct = sum([sc < 0.95 for sc, j in match_list])
                    dups = len(match_list) - num_distinct
                    skips = max(0, dups + min(0, num_distinct - topk))
                    match_list = match_list[skips:]
                    
                    step_size = max(1, len(match_list) // topk)
                    # for sc, j in match_list[step_size - 1::step_size]:
                    for sc, j in match_list[::step_size]:
                        pairs.append((imgs[i], imgs[j]))
                    print(len(match_list[::step_size]), len(match_list), skips)
        
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
