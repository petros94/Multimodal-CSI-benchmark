import torch as torch
import numpy as np

from utils import segment_and_scale


class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, songs, scale=(1, 1)):
        print("Creating MultimodalDataset")
        self.songs = songs
        self.label_mapping = {k: i for i, k in enumerate(list(self.songs.keys()))}
        self.inv_mapping = {i: k for i, k in enumerate(list(self.songs.keys()))}

        self.frames = []
        self.lyrics = []
        self.labels = []
        self.song_names = []
        for song_id, covers in self.songs.items():
            for cover in covers:
                repr = cover['repr']
                lyr = cover['lyrics']
                self.frames.append(segment_and_scale(repr, frame_size=None, scale=scale))
                self.labels.append(self.label_mapping[song_id])
                self.lyrics.append(lyr)
                self.song_names.append(cover['cover_id'])


    def __len__(self):
        return len(self.song_names)

    def __getitem__(self, item):
        return self.frames[item], self.lyrics[item], self.labels[item], self.song_names[item]

    def idx_2_lab(self, idx):
        return self.inv_mapping[idx]

    def filter_per_size(self, songs, frame_size):
        output = {}
        for song_id, covers in songs.items():
            c = []
            for cover in covers:
                if np.array(cover['repr']).shape[-1] > frame_size:
                    c.append(cover)

            if len(c) > 1:
                output[song_id] = c
        return output
