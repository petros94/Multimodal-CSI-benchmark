import json
import torch

from classifier import ThresholdClassifier

from cnn.cnn import from_config as make_audio_model
from text.TextModel import make_model as make_text_model

from MultimodalDataset import MultimodalDataset
from load_songs import from_config
from metrics.ranking_metrics import generate_metrics as generate_ranking_metrics
from metrics.classification_metrics import generate_metrics as generate_classification_metrics
from utils import get_device


def evaluate_test_set(config_path: str = "experiments/evaluation_pretrained.json"):
    
    with open(config_path, "r") as f:
        config = json.load(f)

    test_songs = from_config(config_path=config_path)
    test_set = MultimodalDataset(test_songs)
    audio_model = make_audio_model(config["model"]["config_path"])
    if config["model"]["checkpoint_path"] is not None:
        loc = get_device()
        chk = torch.load(
            config["model"]["checkpoint_path"], map_location=torch.device(loc)
        )
        print("loaded pretrained model")

        audio_model.load_state_dict(chk["model_state_dict"])
        audio_model.eval()

    text_model = make_text_model()

    roc_stats, mean_average_precision, mrr, mr1, pr10 = generate_ranking_metrics(audio_model, text_model, test_set, config['w_audio'])

    print(f"MAP: {mean_average_precision}")
    print(f"MRR: {mrr}")
    print(f"MR1: {mr1}")
    print(f"Pr@10: {pr10}")

    try:
        thr = config["model"]["threshold"]
    except KeyError:
        thr = roc_stats.loc[roc_stats["tpr"] > 0.7, "thr"].iloc[0]

    clf = ThresholdClassifier(audio_model, text_model, thr)
    generate_classification_metrics(clf, test_set, config['w_audio'])