import torch
import numpy as np
from torch.nn import functional as F

def get_device():
    if torch.cuda.is_available():
        print("Using cuda")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("Using metal")
        return "mps"
    else:
        print("Using cpu")
        return "cpu"


def generate_segments(song: np.array, step=400, overlap=0.5):
    """
    Segment a #num_features X num_samples vector into windows.

    Arguments:
        song: np.array or array like of shape num_features x num_samples
        step: the window_size
        overlap: the overlap percentage between windows

    To calculate the time is seconds for each window use the following formula:
    T = step * hop_size / sample_freq
    In the case of mfcc for example, T = step * 512 / 22050

    Returns a python list of shape num_segments X num_features X num_samples
    """
    return [
        song[..., i : step + i]
        for i in np.arange(0, song.shape[-1] - step, int(step * overlap))
    ]


def segment_and_scale(repr, frame_size, scale) -> torch.tensor:
    """
    Take an np.array of shape num_features x num_samples, split into segments and scale to specific size
    in order to create CNN inputs.

    Returns: num_segs X num_channels X num_features X num_samples
    """
    if type(repr) in (torch.tensor, np.ndarray):
        repr = torch.tensor(repr)

        if frame_size is None or repr.size(1) <= frame_size:
            frames = repr.unsqueeze(0)
        else:
            frames = torch.stack(generate_segments(repr, step=frame_size))
        frames = frames.unsqueeze(1)
        frames = F.interpolate(frames, scale_factor=scale)
        return frames

    elif type(repr) == list:
        scaled = [scale_dimensions_to_anchor(repr[0], r) for r in repr]

        # num_channels X num_features X num_samples
        song_repr = torch.stack(scaled)
        # num_segs X num_channels X num_features X num_samples
        if frame_size is not None:
            frames = torch.stack(generate_segments(song_repr, step=frame_size))
        else:
            frames = song_repr.unsqueeze(0)
        frames = F.interpolate(frames, scale_factor=scale)

        assert frames.dim() == 4
        assert frames.size()[1] == len(repr)
        return frames
    else:
        raise ValueError("unsupported repr type")


def scale_dimensions_to_anchor(anchor, repr):
    repr = torch.tensor(repr)
    anchor = torch.tensor(anchor)

    anchor_repr_size = torch.tensor(anchor.size())
    current_repr_size = torch.tensor(repr.size())

    if (torch.all(anchor_repr_size == current_repr_size)):
        # No scale needed
        return repr

    output_size = (anchor_repr_size).tolist()
    repr = repr.unsqueeze(0).unsqueeze(0)
    repr = F.interpolate(repr, size=output_size)
    repr = repr.squeeze(0).squeeze(0)
    return repr