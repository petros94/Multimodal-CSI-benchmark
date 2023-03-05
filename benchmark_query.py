import json
import time
import numpy as np
import torch

from MultimodalDataset import MultimodalDataset
from cnn.cnn import from_config as make_audio_model
from text.TextModel import make_model as make_text_model
from load_songs import from_config
from utils import get_device

w_audio = 0.5

def benchmark_query():
    config_path = "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    print("Loading songs")
    test_songs = from_config(config_path=config_path)
    test_set = MultimodalDataset(test_songs)

    audio_model = make_audio_model(config["model"]["config_path"])
    text_model = make_text_model()
    if config["model"]["checkpoint_path"] is not None:
        loc = get_device()
        chk = torch.load(
            config["model"]["checkpoint_path"], map_location=torch.device(loc)
        )
        print("loaded pretrained model")

        audio_model.load_state_dict(chk["model_state_dict"])
        audio_model.eval()
        audio_model.to(get_device())

    audio_model.eval()
    device = get_device()
    audio_model.type(torch.FloatTensor).to(device)
    audio_embeddings = []
    song_labels = []
    cover_names = []
    with torch.no_grad():
        for frames, song_label, cover_name in zip(test_set.frames, test_set.labels, test_set.song_names):
            x = frames.to(device)
            audio_embeddings.append(audio_model(x))
            song_labels.append(song_label)
            cover_names.append(cover_name)

        audio_embeddings = torch.cat(audio_embeddings, dim=0)
        text_embeddings = text_model.forward(test_set.lyrics)


        query_runtimes = []
        cnn_runtimes = []
        text_runtimes = []
        for frames, song_label, lyrics, cover_name in zip(test_set.frames, test_set.labels, test_set.lyrics, test_set.song_names):
            x = frames.to(device)
            st = time.time()
            aud_emb = audio_model(x)
            cnn_runtimes.append(time.time()-st)


            st_text = time.time()
            text_emb = text_model.forward([lyrics])
            et_text = time.time()
            text_runtimes.append(et_text-st_text)

            st = time.time()
            audio_distance_matrix = torch.cdist(audio_embeddings, aud_emb, p=2)
            text_distance_matrix = torch.cdist(text_embeddings, text_emb, p=2)
            distance_matrix = w_audio * audio_distance_matrix + (1 - w_audio) * text_distance_matrix
            ids = torch.argsort(audio_distance_matrix)
            et = time.time()
            query_runtimes.append(et - st)
        print(f'Mean query time (distance calculation) per song: {np.mean(query_runtimes)}')
        print(f'Mean audio embeddings extraction time per song: {np.mean(cnn_runtimes)}')
        print(f'Mean TF vector extraction time per song: {np.mean(text_runtimes)}')



if __name__ == '__main__':
    benchmark_query()