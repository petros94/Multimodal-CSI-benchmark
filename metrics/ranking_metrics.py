import time

import numpy as np
from sklearn.metrics import (
    roc_curve,
    auc,
    average_precision_score,
    precision_recall_curve
)
import plotly.express as px
import torch
import pandas as pd

from utils import get_device

from MultimodalDataset import MultimodalDataset


def generate_metrics(audio_model, text_model, data_set, w_audio):
    distances, clf_labels, song_labels, cover_names = generate_posteriors_full(audio_model, text_model, data_set, w_audio)
    df = generate_ROC(distances, clf_labels)
    df_prc = generate_PRC(distances, clf_labels)
    map, mrr, mr1, pr10 = ranking_metrics(audio_model, text_model, data_set, w_audio)
    return df, map.item(), mrr, mr1, pr10


def generate_posteriors_full(audio_model, text_model, data_set: MultimodalDataset, w_audio):
    audio_model.eval()
    device = get_device()
    audio_model.type(torch.FloatTensor).to(device)
    audio_embeddings = []
    song_labels = []
    cover_names = []
    distances = []
    clf_labels = []
    runtimes = []
    with torch.no_grad():
        for frames, song_label, cover_name in zip(data_set.frames, data_set.labels, data_set.song_names):
            x = frames.to(device)
            st = time.time()
            audio_embeddings.append(audio_model(x))
            et = time.time()
            runtimes.append(et-st)
            song_labels.append(song_label)
            cover_names.append(cover_name)


        audio_embeddings = torch.cat(audio_embeddings, dim=0)
        audio_distance_matrix = torch.cdist(audio_embeddings, audio_embeddings, p=2)

        text_embeddings = text_model.forward(data_set.lyrics)
        text_distance_matrix = torch.cdist(text_embeddings, text_embeddings, p=2)

        distance_matrix = w_audio*audio_distance_matrix + (1 - w_audio)*text_distance_matrix

        song_labels = torch.tensor(song_labels)
        for id, (d, lab) in enumerate(zip(distance_matrix, song_labels)):
            ids = torch.argwhere(song_labels == lab)
            ids = ids.flatten()
            ids = ids[ids != id]
            pos_dist = d[ids]
            distances.append(pos_dist)
            clf_labels.extend([1] * pos_dist.size()[0])

            ids = torch.argwhere(song_labels != lab)
            ids = ids.flatten()
            neg_dist = d[ids]
            distances.append(neg_dist)
            clf_labels.extend([0] * neg_dist.size()[0])


        distances, clf_labels = torch.cat(distances).cpu().numpy(), np.array(clf_labels)
        return distances, clf_labels, song_labels, cover_names


def generate_ROC(distances: np.ndarray, clf_labels: np.ndarray):
    permute_ids = np.random.permutation(len(distances))
    sample_distances = distances[permute_ids][:100000]
    sample_clf_labels = clf_labels[permute_ids][:100000]

    fpr, tpr, thresholds = roc_curve(sample_clf_labels, 2 - sample_distances)
    df = pd.DataFrame({"tpr": tpr, "fpr": fpr, "thr": thresholds})
    roc_auc = auc(fpr, tpr)

    fig = px.area(
        data_frame=df,
        x="fpr",
        y="tpr",
        title=f"ROC Curve (AUC={roc_auc:.4f})",
        labels=dict(x="False Positive Rate", y="True Positive Rate"),
        hover_data=["thr"],
        width=700,
        height=500,
    )
    fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain="domain")
    fig.show()
    return df


def calc_MRR(distance_matrix: torch.Tensor, labels: torch.Tensor):
    # Size N x N
    rr = []

    for d, lab in zip(distance_matrix, labels):
        sorted_ids_by_dist = d.argsort()
        sorted_labels_by_dist = labels[sorted_ids_by_dist]

        # MRR
        rank = (sorted_labels_by_dist[1:] == lab).nonzero()[0].item() + 1
        rr.append(1 / rank)

    mrr = np.mean(rr)
    return mrr

def calc_MAP(array2d, version, data_set):
    MAP, top10, rank1 = 0, 0, 0

    for d, lab in zip(array2d, version):
        per_top10, per_rank1, per_MAP = 0, 0, 0
        version_cnt = torch.tensor(0)

        sorted_ids_by_dist = d.argsort()
        sorted_labels_by_dist = version[sorted_ids_by_dist][1:]

        ids = torch.argwhere(sorted_labels_by_dist == lab)
        for id in ids:
            version_cnt += 1
            per_MAP += version_cnt / (id + 1)
            if per_rank1 == 0:
                per_rank1 = id + 1
            if id < 10:
                per_top10 += 1

        per_MAP /= 1 if version_cnt < 0.0001 else version_cnt
        MAP += per_MAP
        top10 += per_top10
        rank1 += per_rank1

    return MAP / float(len(array2d)), top10 / float(len(array2d)) / 10, rank1 / float(len(array2d))


def ranking_metrics(audio_model, text_model, data_set, w_audio):
    audio_model.eval()
    device = get_device()
    audio_model.type(torch.FloatTensor).to(device)
    with torch.no_grad():
        audio_embeddings = []
        labels = torch.tensor(data_set.labels).to(device)
        for frames, label in zip(data_set.frames, data_set.labels):
            x = frames.to(device)
            audio_embeddings.append(audio_model(x))

        audio_embeddings = torch.cat(audio_embeddings, dim=0)
        audio_distance_matrix = torch.cdist(audio_embeddings, audio_embeddings, p=2)

        text_embeddings = text_model.forward(data_set.lyrics)
        text_distance_matrix = torch.cdist(text_embeddings, text_embeddings, p=2)

        distance_matrix = w_audio * audio_distance_matrix + (1 - w_audio) * text_distance_matrix

        map, prk, mr1 = calc_MAP(distance_matrix, labels, data_set)
        mrr = calc_MRR(distance_matrix, labels)


        return map, mrr, mr1, prk


def generate_PRC(distances: np.ndarray, clf_labels: np.ndarray):
    permute_ids = np.random.permutation(len(distances))
    sample_distances = distances[permute_ids][:100000]
    sample_clf_labels = clf_labels[permute_ids][:100000]

    # Use the precision_recall_curve function to get the precision, recall, and thresholds arrays
    precision, recall, thresholds = precision_recall_curve(sample_clf_labels, 2 - sample_distances)
    df = pd.DataFrame({"precision": precision[:-1], "recall": recall[:-1], "thr": thresholds})
    # Compute the average precision score
    ap = average_precision_score(sample_clf_labels, 2 - sample_distances)

    # Create a Plotly area plot using the precision, recall, and thresholds arrays
    fig = px.line(
        data_frame=df,
        x="recall",
        y="precision",
        title=f"Precision-Recall Curve (AP={ap:.4f})",
        labels=dict(x="Recall", y="Precision"),
        hover_data=["thr"],
        width=700,
        height=500,
    )

    # Update the y-axis to have the same scale as the x-axis
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    # Constrain the x-axis to the [0, 1] domain
    fig.update_xaxes(constrain="domain")

    # Display the plot
    fig.show()
    return df