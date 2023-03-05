import os
import json

import numpy as np
import scipy.io


def from_config(config_path: str = "songbase/config.json"):
    with open(config_path, "r") as f:
        config = json.load(f)

    test_songs_audio = {}
    try:
        for dataset in config["test_datasets_audio"]:
            songs = load_songs(
                type=dataset["type"],
                songs_dir=dataset["path"]
            )
            test_songs_audio.update(songs)
    except KeyError as e:
        print("No audio test datasets supplied.")
        print(e)

    test_songs_lyrics = {}

    try:
        for dataset in config["test_datasets_lyrics"]:
            lyrics = load_lyrics(song_dir=dataset["path"])
            test_songs_lyrics.update(lyrics)
    except KeyError:
        print("No lyrics test datasets supplied.")

    return merge_modalities(test_songs_audio, test_songs_lyrics)


def load_songs(type="shs100k", songs_dir=["hpcp/"], features=["hpcp"]):
    """
    Load the song database in JSON format.

    Example:
    {
        "120345 (song_id)": [
            {
                "song_id": "120345",
                "cover_id": "454444",
                "repr": [[3.43, 2.34, 5.55, ...], [4.22, 0.45, 3.44], ...]  #num_features X num_samples
            },
            ...
        ]
    }

    Arguments:
        type: ["covers1000", "covers80"] the dataset type
    """
    if type == "shs100k":
        return load_songs_shs100k(songs_dir, features)
    elif type == "coversshs":
        return load_songs_coversshs(songs_dir, features)
    else:
        raise ValueError("'type' must be one of ['covers1000', 'covers80']")


def load_songs_shs100k(songs_dir, features=['hpcp']):
    songs = {}
    for song_dir, feature in zip(songs_dir, features):
        songs[feature] = {}
        origin_path = song_dir
        entries = os.listdir(origin_path)

        for item in entries:
            song_id, cover_id = item.split(".npy")[0].split("_")
            repr = np.transpose(np.load(os.path.join(origin_path, item)))

            if song_id not in songs[feature].keys():
                songs[feature][song_id] = []

            songs[feature][song_id].append(
                {"song_id": song_id, "cover_id": cover_id, "repr": repr}
            )
    return merge_song_representations(songs)


def load_songs_coversshs(songs_dir=["hpcps80/"], features=["hpcp"]):
    songs = {}
    skipped_counter = 0
    for song_dir, feature in zip(songs_dir, features):
        songs[feature] = {}
        origin_path = song_dir
        entries = os.listdir(origin_path)

        if feature == "mfcc":
            mat_feature = "XMFCC"
        elif feature == "hpcp":
            mat_feature = "XHPCP"
        elif feature == "cens":
            mat_feature = "XCENS"
        elif feature == "wav":
            mat_feature = "XWAV"

        for dir in entries:
            if not os.path.isdir(os.path.join(origin_path, dir)):
                continue
            subdir = os.listdir(origin_path + dir)

            if len(subdir) <= 1:
                skipped_counter += 1
                print(
                    f"Warning found song with no covers: {origin_path + dir}, skipping..."
                )
                continue

            songs[feature][dir] = []

            for song in subdir:
                if not song.endswith('.mat'):
                    continue
                song_id = dir
                cover_id = song
                mat = scipy.io.loadmat(origin_path + dir + "/" + song)
                repr = mat[mat_feature]  # No need to normalize since already normalized
                repr = np.transpose(np.array(repr))
                songs[feature][dir].append(
                    {"song_id": song_id, "cover_id": cover_id, "repr": repr}
                )

    print(f"Total: {len(songs[features[0]])}, skipped: {skipped_counter}")
    return merge_song_representations(songs)



def merge_song_representations(songs):
    """Merge song representations.

    Args:
        songs (dict): songs in the following format:
        {
            'mfcc': {
                'song_id': [
                    {"song_id": 'song_id', "cover_id": 'cover_id_1', "repr": repr_11},
                    {"song_id": 'song_id', "cover_id": 'cover_id_2', "repr": repr_12},
                    {"song_id": 'song_id', "cover_id": 'cover_id_3', "repr": repr_13}
                ],
                'song_id_2': [...]
                ...
            },
            'hpcp': {
                'song_id': [
                    {"song_id": 'song_id', "cover_id": 'cover_id_1', "repr": repr_21},
                    {"song_id": 'song_id', "cover_id": 'cover_id_2', "repr": repr_22},
                    {"song_id": 'song_id', "cover_id": 'cover_id_3', "repr": repr_23}
                ],
                'song_id_2': [...],
                ...
            }
        }

    Returns:
        concatenated song dict:
        {
            'song_id': [
                    {"song_id": 'song_id', "cover_id": 'cover_id_1', "repr": [repr_11, repr_21]},
                    {"song_id": 'song_id', "cover_id": 'cover_id_2', "repr": [repr_12, repr_22]},
                    {"song_id": 'song_id', "cover_id": 'cover_id_3', "repr": [repr_13, repr_23]}
            ],
            'song_id_2': [...]
            ...
        }
    """
    features = list(songs.keys())
    anchor_feat = features[0]
    songs_ids = list(songs[anchor_feat].keys())
    songs_concatenated_features = {}
    for id in songs_ids:
        songs_concatenated_features[id] = []

        covers = [songs[feature][id] for feature in features]
        for feats in zip(*covers):
            song_id = feats[0]["song_id"]
            cover_id = feats[0]["cover_id"]
            repr = [r["repr"] for r in feats]
            songs_concatenated_features[id].append(
                {"song_id": song_id, "cover_id": cover_id, "repr": repr}
            )
    return songs_concatenated_features



def load_lyrics(song_dir):
    song_lyrics = {}
    for song in os.listdir(song_dir):
        if not os.path.isdir(os.path.join(song_dir, song)):
            continue

        song_lyrics[song] = []
        for cover in os.listdir(os.path.join(song_dir, song)):
            if not cover.endswith(".txt"):
                continue

            with open(os.path.join(song_dir, song, cover)) as f:
                lyrics = "".join(f.readlines())
                lyrics = lyrics.strip()

                song_id = song
                cover_id = cover

                song_lyrics[song].append({"song_id": song_id, "cover_id": cover_id, "lyrics": lyrics})

    return song_lyrics


def merge_modalities(song_audio, song_lyrics):
    song_modalities = {}

    for song_key in song_audio.keys():
        song_modalities[song_key] = []

        covers_audio = song_audio[song_key]
        covers_lyrics = song_lyrics[song_key]
        for cover in covers_audio:
            cover_id = cover['cover_id'].split(".")[0]
            repr = cover['repr']
            lyrics = ""

            for it in covers_lyrics:
                if it['cover_id'].split(".")[0] == cover_id:
                    lyrics = it['lyrics']
                    break

            song_modalities[song_key].append({
                'song_id': song_key,
                'cover_id': cover_id,
                'repr': repr,
                'lyrics': lyrics
            })
    return song_modalities




