from collections.abc import Sequence
from typing import Iterator

import av
import numpy as np


class _AVVideo(Sequence[av.VideoFrame]):
    """
    Class to access PyAV video frames by index.

    Note that in general, videos could have variable frame rates or the timing metadata stored inside of them is sloppy.
    Right now, the behavior is to crash so that we can investigate the issue and decided whether to support these
    kinds of files.

    Example Usage:

    with AVVideo(path) as video:
        print(len(video))
        frame_a = video[0] # get the first frame
        frame_b = video[12]  # get frame 12
        frame_c = video[len(video) - 1] # get the last frame

        for frame in video:  # iterate over all frames
            pass

    Generally speaking, sequential access is faster than random access since random access may require seeking and decoding
    frames that will be discarded.
    """

    def __init__(
        self,
        path,
        video_stream_id=0,
        thread_type="SLICE",
        thread_count=0,
    ):
        """
        Initialize AVVideo

        :param path: path to video file
        :param video_stream_id: id of video in container (i.e. 0 is the first video stream)
        :param thread_type: 'SLICE' or 'FRAME', or 'AUTO'
        see https://pyav.basswood-io.com/docs/develop/api/codec.html#av.codec.context.ThreadType,
        and https://pyav.basswood-io.com/docs/stable/cookbook/basics.html#threading
        :param thread_count: number of threads to use (0 is auto)

        The best thread type to use depends on the way the video is encoded and your access pattern.
        The only situation I have found where 'SLICE' threading works better is when you
        are accessing random frames (without any temporal window) on a purely IFrame encoded video.
        In all other situations 'AUTO' (which seems to be both 'SLICE' and 'FRAME') works best.
        """

        container: av.container.InputContainer = av.open(path)
        stream: av.video.stream.VideoStream = container.streams.video[video_stream_id]
        stream.thread_type = thread_type
        stream.thread_count = thread_count

        _AVVideo._verify_timing(stream)

        self._container = container
        self._stream = stream
        self._next_frame_idx = 0
        self._iterator = self._create_iterator()

    @staticmethod
    def _verify_timing(stream):
        """
        Verify that the stream metadata satisfies our assumptions about timing
        see https://pyav.basswood-io.com/docs/stable/api/time.html
        """

        if stream.start_time != 0:
            raise ValueError("video stream starts at an offset")

        if stream.frames == 0:
            raise ValueError("unknown number of frames in video")

        # each frame has an integer presentation time stamp (pts) which counts the number of time_base seconds
        # the base_rate should give the frames per second
        # if we calculate how many pts are in each frame, this should come out to an integer
        # see
        pts_per_frame = 1 / (stream.base_rate * stream.time_base)
        if pts_per_frame.denominator != 1:
            raise ValueError("pts per frame is not an integer")

        # duration in seconds == number of frames / fps
        if stream.duration * stream.time_base != stream.frames / stream.base_rate:
            raise ValueError(
                "duration of video in seconds is inconsistent with the number of frames"
            )

    def close(self):
        self._container.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()  # in case user forgets to explicitly close

    def __len__(self):
        return self._stream.frames

    @staticmethod
    def _create_iterator_static(
        container: av.container.InputContainer,
        stream: av.video.VideoStream,
        next_frame_idx: int,
    ) -> Iterator[av.video.frame.VideoFrame]:
        # this method is static to avoid circular references with 'self' inside self._iterator (which apparently causes a memory leak with av resources)
        for frame in container.decode(stream.index):
            frame: av.video.frame.VideoFrame

            # frame index = seconds * fps
            frame_idx = (frame.pts * frame.time_base) * stream.base_rate

            if frame_idx != round(frame_idx):
                raise ValueError("video frame timing is inconsistent")
            frame_idx = round(frame_idx)

            if frame_idx > next_frame_idx:
                raise ValueError("video is missing frames")

            # might need to skip some frames after a seek
            if frame_idx < next_frame_idx:
                continue

            yield frame
            next_frame_idx = frame_idx + 1

    def _create_iterator(self):
        return _AVVideo._create_iterator_static(
            container=self._container,
            stream=self._stream,
            next_frame_idx=self._next_frame_idx,
        )

    def _seek(self, frame_idx):
        pts_offset = round(frame_idx / self._stream.base_rate / self._stream.time_base)
        self._container.seek(
            offset=pts_offset, backward=True, any_frame=False, stream=self._stream
        )
        self._next_frame_idx = frame_idx
        self._iterator = self._create_iterator()

    def _read(self):
        frame = next(self._iterator)
        self._next_frame_idx += 1
        return frame

    def __getitem__(self, frame_idx: int):
        if not 0 <= frame_idx < len(self):
            raise IndexError

        # very valuable to not seek unless it's necessary,
        if frame_idx != self._next_frame_idx:
            self._seek(frame_idx)

        return self._read()


class Video(Sequence[np.ndarray]):
    """
    Class to access video frames by index. Frames are returned as numpy arrays.

    The format of the output numpy arrays can be configured.

    Example Usage:

    with Video(path, width=100, height=100) as video:
        assert video[0].shape[:2] == (100, 100)  # True

        for frame in video:  # iterate over all frames
            pass
    """

    def __init__(
        self,
        path,
        pix_fmt="rgb24",
        width=None,
        height=None,
        thread_type="SLICE",
        thread_count=0,
    ):
        """
        Initialize Video

        :param path: path to video file
        :param pix_fmt: pixel format when converting to numpy array (default rgb24, which is 8 bits per channel)
        see https://pyav.basswood-io.com/docs/stable/api/video.html#av.video.format.VideoFormat
        :param width: output width (None for same as video)
        :param height: output height (None for same as video)
        :param thread_type: thread type argument to pyav stream (see AVVideo docs)
        :param thread_count: thread count argument to pyav stream (see AVVideo docs)
        """

        self._av_video = _AVVideo(
            path, thread_type=thread_type, thread_count=thread_count
        )
        self._pix_fmt = pix_fmt
        self._width = width
        self._height = height

    def close(self):
        self._av_video.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __len__(self):
        return len(self._av_video)

    def __getitem__(self, item):
        frame = self._av_video[item]
        return frame.to_ndarray(
            format=self._pix_fmt, width=self._width, height=self._height
        )


def _benchmark():
    from itertools import product
    from pathlib import Path
    from tqdm import tqdm

    # video_dir = "D:/NFL"
    video_dir = r"\\datamtl\RAID\NFL\videos"
    filenames = [
        "2024.08.08.NEP vs Panthers.mp4",
        # "2024.09.22.H2.ATL vs Chiefs.mp4",
        # "2024.10.13.BAL vs Commanders.mp4",
    ]

    path_list = [Path(video_dir) / fn for fn in filenames]
    thread_type_list = ["SLICE", "FRAME", "AUTO"]
    # 0 for sequential, > 0 for size of temporal window of random access
    access_pattern_list = [0, 1, 3]
    n_iters = 100

    for path, thread_type, access_pattern in product(
        path_list, thread_type_list, access_pattern_list
    ):
        with Video(path, thread_type=thread_type) as video:
            indices = np.arange(len(video))
            if access_pattern > 0:
                temporal_window = access_pattern
                indices = indices[: len(video) // temporal_window * temporal_window]
                indices = indices.reshape(-1, temporal_window)
                indices = indices[np.random.permutation(indices.shape[0]), :]
                indices = indices.flatten()

            indices = indices[-n_iters:]

            print("file:", path)
            print("thread type:", thread_type)
            print(
                "access pattern:",
                (
                    f"Random (Window Size {access_pattern})"
                    if access_pattern > 0
                    else "Sequential"
                ),
            )
            for idx in tqdm(indices):
                _ = video[idx]
            print()


def _check_frame():
    import PIL.Image

    image_path = r"\\datamtl\RAID\NFL\datasets\data_to_annotate\2024.10.21.H1.TBB vs Ravens\2024.10.21.H1.TBB vs Ravens.mp4_0065760.jpg"
    frame_idx = 65760
    video_path = r"\\datamtl\RAID\NFL\videos\2024.10.21.H1.TBB vs Ravens.mp4"

    PIL.Image.open(image_path).show("target")
    with _AVVideo(video_path) as video:
        video[frame_idx].to_image().show("actual")


def _check_memory():
    """
    Load random frames from random videos in a directory. Check your system memory to identify any memory issues
    """
    import pathlib
    import random
    from tqdm import tqdm
    from slomodata.video import Video as Video2

    # video_paths = list(pathlib.Path(r"\\datamtl\RAID\Datasets\NFL").glob("*.mp4"))
    video_paths = list(pathlib.Path(r"D:\Recording vs Dirty Feed").glob("*.mp4"))

    temporal_size = 3
    n_iters = 1_000_000
    for _ in tqdm(range(n_iters)):
        video_idx = random.randint(0, len(video_paths) - 1)

        with Video(video_paths[video_idx]) as video:
            frame_idx = random.randint(0, len(video) - 1)
            _images = [
                video[(frame_idx + i) % len(video)]
                for i in range(frame_idx, frame_idx + temporal_size)
            ]


if __name__ == "__main__":
    # _benchmark()
    # _check_frame()
    _check_memory()
