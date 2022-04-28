import os
import wandb
import argparse
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

import pytorch_lightning as pl

from typing import Callable, Iterable, Union

from transformers import BertTokenizer, BertModel
from transformers import HfArgumentParser
from transformers.hf_argparser import DataClassType

from torch.nn import CosineSimilarity
from scipy.spatial import ConvexHull
from sklearn.manifold import TSNE
from collections import defaultdict
from tqdm import tqdm
from pytorch_lightning.loggers import WandbLogger

from random import random

from Multilingual_CLIP.src import multilingual_clip
from src.dataset import get_datamodule
from src.model import FineTuner
from src.parse_args import BasicArguments


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dim_reducer = TSNE(n_components=2)


def visualize_layerwise_embeddings(hidden_states, label, cluster, colors):

    if not os.path.exists("plotted_output"):
        os.makedirs("plotted_output")
    
    layer_embeds = hidden_states
        
    layer_dim_reduced_embeds = dim_reducer.fit_transform(layer_embeds.cpu().numpy())
    
    df = pd.DataFrame.from_dict({'x':layer_dim_reduced_embeds[:,0],'y':layer_dim_reduced_embeds[:,1],'label':label, 'c': colors[label], 'cluster': cluster})
    return df, layer_dim_reduced_embeds[:,0].sum()/layer_dim_reduced_embeds.shape[0], layer_dim_reduced_embeds[:,1].sum()/layer_dim_reduced_embeds[:,1].shape[0]



def centroid(dps, mtype):
    # return torch.stack(dps[mtype], dim=0).sum(dim=0).sum(dim=0) / len(dps[mtype])   
    return torch.stack(dps[mtype], dim=0)


def scatter_plot(model_hidden_states, ln1, ln2, colors):
    # Plot of the embedding space
    dfs, cen_xs, cen_ys = [], [], []

    fig, ax = plt.subplots(1, figsize=(8,8))
    for i, (k, itm) in enumerate(model_hidden_states.items()):
        hidden_states = torch.cat(itm, dim=0)

        df, cen_x, cen_y = visualize_layerwise_embeddings(hidden_states=hidden_states, label=k, cluster=i, colors=colors)
        plt.scatter(df.x, df.y, c=df.c, alpha = 0.6, s=10, label=k)
        dfs.append(df); cen_xs.append(cen_x); cen_ys.append(cen_y)

    df = pd.concat(dfs, ignore_index=True, sort=False)

    # plot data
    # plot centers
    plt.scatter(cen_xs, cen_ys, marker='^', c=colors.values(), s=70)
    # draw enclosure
    for i in df.cluster.unique():
        points = df[df.cluster == i][['x', 'y']].values
        # get convex hull
        hull = ConvexHull(points)
        # get x and y coordinates
        # repeat last point to close the polygon
        x_hull = np.append(points[hull.vertices,0],
                           points[hull.vertices,0][0])
        y_hull = np.append(points[hull.vertices,1],
                           points[hull.vertices,1][0])
        # plot shape
        plt.fill(x_hull, y_hull, alpha=0.3, c=list(colors.values())[i])
        
    # plt.xlim(0,200)
    # plt.ylim(0,200)

    # sns.scatterplot(data=df,x='x',y='y',hue='label')
    plt.legend()    
    plt.savefig(f'plotted_output/emb_mbert_mclip_{ln1}_{ln2}.png',format='png',pad_inches=0)


def parse_args(dataclass_types: Union[DataClassType, Iterable[DataClassType]]):
    parser = HfArgumentParser(dataclass_types)
    args_in_dataclasses_and_extra_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    args_in_dataclasses, extra_args = args_in_dataclasses_and_extra_args[:-1], args_in_dataclasses_and_extra_args[-1]

    assert not extra_args, f"Unknown arguments: {extra_args}"

    args = argparse.Namespace(**{k: v for args in args_in_dataclasses for k, v in args.__dict__.items()})

    return args


def main():

    args = parse_args(BasicArguments)

    colors = {
        f"mbert_{args.ln1}": "red",
        f"mbert_{args.ln2}": "orange",
        f"mclip_{args.ln1}": "blue",
        f"mclip_{args.ln2}": "green"
    }

    dps = {}

    wandb_logger = WandbLogger(name="mBERT-mCLIP", project="multilingual-multimodal", offline=True)

    # initialize the lightening trainer scheme
    # NOTE: the trainer seems to take a lot of time to initialize the logger.
    # Thus, we disable the logger to make the process run faster
    trainer = pl.Trainer(gpus=1, precision=16, logger=False)

    # model specification
    mbert_model = BertModel.from_pretrained("bert-base-multilingual-cased")
    model = FineTuner(model=mbert_model, ln1=args.ln1, ln2=args.ln2, mtype="mbert")
    trainer.test(model=model, datamodule=get_datamodule("mbert", ln1=args.ln1, ln2=args.ln2))
    for k, l in model.stats.items():
        dps[f"mbert_{k}"] = l

    mclip_model = multilingual_clip.load_model('M-BERT-Base-69')
    model = FineTuner(model=mclip_model, ln1=args.ln1, ln2=args.ln2, mtype="mclip")
    trainer.test(model=model, datamodule=get_datamodule("mclip", ln1=args.ln1, ln2=args.ln2))
    for k, l in model.stats.items():
        dps[f"mclip_{k}"] = l

    # avg output the embedding
    # model_hidden_states["mbert_en"].append(mbert_en.unsqueeze(0).sum(dim=1)/mbert_en.shape[1])
    # model_hidden_states["mbert_ru"].append(mbert_ru.unsqueeze(0).sum(dim=1)/mbert_ru.shape[1])
    # model_hidden_states["mclip_en"].append(mclip_en.unsqueeze(0).sum(dim=1)/mclip_en.shape[1])
    # model_hidden_states["mclip_ru"].append(mclip_ru.unsqueeze(0).sum(dim=1)/mclip_ru.shape[1])

    mbert_ln1_cen = centroid(dps, f"mbert_{args.ln1}").squeeze(dim=1)
    mbert_ln2_cen = centroid(dps, f"mbert_{args.ln2}").squeeze(dim=1)
    mclip_ln1_cen = centroid(dps, f"mclip_{args.ln1}").squeeze(dim=1)
    mclip_ln2_cen = centroid(dps, f"mclip_{args.ln2}").squeeze(dim=1)
    cos = CosineSimilarity(dim=1)
    N, _ = mbert_ln1_cen.shape
    print(f"Between {args.ln1} and {args.ln2} mBERT similarity score: {cos(mbert_ln1_cen, mbert_ln2_cen).sum()/N}; mCLIP similarity score: {cos(mclip_ln1_cen, mclip_ln2_cen).sum()/N}")
    print(f"similarity between mBERT and mCLIP {args.ln1} similarity: {cos(mbert_ln1_cen, mclip_ln1_cen).sum()/N}; {args.ln2} similarity: {cos(mbert_ln2_cen, mclip_ln2_cen).sum()/N}")

    # scatter_plot(dps, args.ln1, args.ln2, colors)

if __name__ == "__main__":
    main()