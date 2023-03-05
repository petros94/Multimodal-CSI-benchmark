import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
    accuracy_score,
)
import matplotlib.pyplot as plt
import torch
import pandas as pd

from utils import get_device

from MultimodalDataset import MultimodalDataset
from classifier import ThresholdClassifier


def generate_metrics(clf, data_set, w_audio):
    return generate_metrics_full(clf, data_set, w_audio)


def generate_metrics_full(clf: ThresholdClassifier, data_set: MultimodalDataset, w_audio):
    clf.audio_model.eval()
    device = get_device()
    clf.audio_model.type(torch.FloatTensor).to(device)
    audio_embeddings = []
    y_pred = []
    y_true = []
    with torch.no_grad():
        for frames, label in zip(data_set.frames, data_set.labels):
            x = frames.to(device)
            audio_embeddings.append(clf.audio_model(x))

        audio_embeddings = torch.cat(audio_embeddings, dim=0)
        audio_distance_matrix = torch.cdist(audio_embeddings, audio_embeddings, p=2)

        text_embeddings = clf.text_model.forward(data_set.lyrics)
        text_distance_matrix = torch.cdist(text_embeddings, text_embeddings, p=2)

        distance_matrix = w_audio * audio_distance_matrix + (1 - w_audio) * text_distance_matrix

        song_labels = torch.tensor(data_set.labels)

        for id, (d, lab) in enumerate(zip(distance_matrix, song_labels)):
            ids = torch.argwhere(song_labels == lab)
            ids = ids.flatten()
            ids = ids[ids != id]
            pos_dist = d[ids]

            inv = 2 - pos_dist
            pos_preds = (inv > clf.D) * 1
            pos_preds = pos_preds.cpu().tolist()

            y_pred.extend(pos_preds)
            y_true.extend([1] * len(pos_preds))

            ids = torch.argwhere(song_labels != lab)
            ids = ids.flatten()
            neg_dist = d[ids]
            inv = 2 - neg_dist
            neg_preds = (inv > clf.D) * 1
            neg_preds = neg_preds.cpu().tolist()
            y_pred.extend(neg_preds)
            y_true.extend([0] * len(neg_preds))

        
        return generate_metrics_bare(y_true, y_pred)


def generate_metrics_bare(y_true, y_pred):
    permute_ids = np.random.permutation(len(y_true))
    sample_y_true = np.array(y_true)[permute_ids][:100000]
    sample_y_pred = np.array(y_pred)[permute_ids][:100000]

    pr, rec, f1, sup = precision_recall_fscore_support(sample_y_true, sample_y_pred)
    acc = accuracy_score(sample_y_true, sample_y_pred)
    df = pd.DataFrame({"pre": pr, "rec": rec, "f1": f1, "sup": sup})
    ConfusionMatrixDisplay.from_predictions(sample_y_true, sample_y_pred)

    print(f"Accuracy is: {acc}")
    print(df)
    plt.show()
    return df