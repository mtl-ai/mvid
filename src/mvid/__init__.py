from collections.abc import Sequence
from typing import Iterator, Literal

import av
import numpy as np

AVThreadType = Literal["SLICE", "FRAME", "AUTO"]


class AVVideo(Sequence[av.VideoFrame]):
    """
    Provides random and sequential access to video frames using PyAV.

    This class exposes a simple Pythonic sequence interface for reading frames by index,
    iterating through frames, and querying the total number of frames.
    It assumes a video with a stable frame rate. Videos with variable
    frame rates or inconsistent timing metadata may raise errors. This is
    intentional so such cases can be inspected and future support evaluated.

    Sequential access is generally faster than random access because random access may
    require seeking and decoding intermediate frames that are ultimately discarded.

    Example usage:

    ```python
    with AVVideo(path) as video:
        print(len(video))                 # total number of frames
        frame_a = video[0]                # first frame
        frame_b = video[12]               # frame 12
        frame_c = video[len(video) - 1]   # last frame

        for frame in video:               # sequential iteration
            pass
    ```
    """

    def __init__(
        self,
        path,
        video_stream_id=0,
        thread_type: AVThreadType = "SLICE",
        thread_count=0,
    ):
        """
        Initialize AVVideo class.

        :param path: path to video file
        :param video_stream_id: id of video in container (i.e. 0 is the first video stream)
        :param thread_type: 'SLICE' or 'FRAME', or 'AUTO'
        see https://pyav.basswood-io.com/docs/develop/api/codec.html#av.codec.context.ThreadType,
        and https://pyav.basswood-io.com/docs/stable/cookbook/basics.html#threading
        :param thread_count: number of threads to use (0 is auto)

        The best thread type to use depends on the way the video is encoded and your access pattern.
        """

        container: av.container.InputContainer = av.open(path)
        stream: av.video.stream.VideoStream = container.streams.video[video_stream_id]
        stream.thread_type = thread_type
        stream.thread_count = thread_count

        AVVideo._verify_timing(stream)

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
            raise ValueError("Video stream starts at an offset")

        if stream.frames == 0:
            raise ValueError("Unknown number of frames in video")

        # The stream time_base gives the number of seconds per 'tick'.
        # Each frame has presentation time stamp (PTS) which counts in 'ticks'.
        # The stream base_rate should give the frames per second (FPS) of the video.
        # (NOTE: perhaps guessed_rate would be a good choice to use instead).
        # If we calculate how many pts (ticks) are in a frame, this should be an integer.
        # 1 / ticks_per_frame = frames_per_second * seconds_per_tick
        pts_per_frame = 1 / (stream.base_rate * stream.time_base)
        if pts_per_frame.denominator != 1:
            raise ValueError(
                f"PTS per frame ({float(pts_per_frame)}) is not an integer "
            )

        # duration in seconds == number of frames / fps
        if stream.duration * stream.time_base != stream.frames / stream.base_rate:
            raise ValueError(
                f"Duration of video in seconds is inconsistent with the number of frames"
            )

    def close(self):
        self._container.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __len__(self):
        return self._stream.frames

    @staticmethod
    def _create_iterator_static(
        container: av.container.InputContainer,
        stream: av.video.VideoStream,
        next_frame_idx: int,
    ) -> Iterator[av.video.frame.VideoFrame]:
        # this method is static to avoid circular references with 'self' inside self._iterator
        # (which can cause slow leak of resources if the gc doesn't handle it fast enough)
        for frame in container.decode(stream.index):
            frame: av.video.frame.VideoFrame

            # frame index = (ticks * seconds_per_tick) * fps
            frame_idx = (frame.pts * frame.time_base) * stream.base_rate

            if frame_idx != round(frame_idx):
                raise ValueError(
                    f"Video frame index is not an integer ({float(frame_idx)})"
                )
            frame_idx = round(frame_idx)

            if frame_idx > next_frame_idx:
                raise ValueError(f"Video is missing frame {next_frame_idx}")

            # might need to skip some frames after a seek
            if frame_idx < next_frame_idx:
                continue

            yield frame
            next_frame_idx = frame_idx + 1

    def _create_iterator(self):
        return AVVideo._create_iterator_static(
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
    Provides random and sequential access to videos. Frames are returned as NumPy arrays.

    This class exposes a simple Pythonic sequence interface for reading frames by index,
    iterating through frames, and querying the total number of frames.

    Example usage:

    ```python
    with Video(path) as video:
        print(len(video))                 # total number of frames
        frame_a = video[0]                # first frame
        frame_b = video[12]               # frame 12
        frame_c = video[len(video) - 1]   # last frame

        for frame in video:               # sequential iteration
            pass
    ```
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

        self._av_video = AVVideo(
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
