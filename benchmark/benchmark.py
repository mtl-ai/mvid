from itertools import product
import sys

import numpy as np
from tqdm import tqdm

from mvid import AVVideo, Video

if __name__ == "__main__":
    video_path = sys.argv[1]
    tqdm.write(f"Compare with `ffmpeg -i {video_path} -f null -`")
    tqdm.write("")

    class_list = [AVVideo, Video]
    thread_type_list = ["AUTO", "SLICE", "FRAME"]
    thread_count_list = [0]  # [0, 1, 5]

    access_pattern_list = [
        0,
        1,
        5,
    ]  # number of sequence frames to access before random skip

    n_frames = 500

    tqdm.write(f"Video Path: {video_path}")
    for t_type, t_count, window, cls in product(
        thread_type_list, thread_count_list, access_pattern_list, class_list
    ):
        with cls(video_path, thread_type=t_type, thread_count=t_count) as video:
            indices = np.arange(len(video))

            if window > 0:
                indices = indices[: len(video) // window * window]
                indices = indices.reshape(-1, window)
                indices = indices[np.random.permutation(indices.shape[0]), :]
                indices = indices.flatten()

            indices = indices[:n_frames]

            tqdm.write(f"Class: {cls.__name__}")
            tqdm.write(f"Thread Type: {t_type}")
            tqdm.write(f"Thread Count: {t_count}")
            tqdm.write(
                f"Access Pattern: {f'Random Window {window}' if window > 0 else 'Sequential'}"
            )

            for idx in tqdm(indices):
                _ = video[idx]
