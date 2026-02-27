from collections.abc import Sequence
from typing import Iterator, Literal

import av
import numpy as np


class AVVideo(Sequence[av.VideoFrame]):
    """
    This is the "raw" PyAV version of the Video class. It returns PyAV Frame objects.

    See Video docs for more information about usage.

    This class takes care of all the necessary seeking and bookkeeping.

    The main idea is to seek only when necessary and to decode all the frames until we reach the target frame index.
    """

    def __init__(
        self,
        path,
        video_stream_id=0,
        thread_type="SLICE",
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

        if thread_type not in ('SLICE', 'FRAME', 'AUTO'):
            raise ValueError(f"thread_type '{thread_type}' is not 'SLICE', 'FRAME', or 'AUTO'")

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
            raise ValueError("Unknown number of frames in the video file")

        # The stream time_base gives the number of seconds per 'tick'.
        # Each frame has presentation time stamp (PTS) which counts in 'ticks'.
        # The stream base_rate should give the frames per second (FPS) of the video.
        # (NOTE: perhaps guessed_rate would be a good choice to use instead).
        # If we calculate how many pts (ticks) are in a frame, this should be an integer.
        # 1 / ticks_per_frame = frames_per_second * seconds_per_tick
        pts_per_frame = 1 / (stream.base_rate * stream.time_base)
        if pts_per_frame.denominator != 1:
            raise ValueError(
                f"PTS per frame ({float(pts_per_frame)}) is not an integer for this video stream, check your file"
            )

        # duration in seconds == number of frames / fps
        if stream.duration * stream.time_base != stream.frames / stream.base_rate:
            raise ValueError(
                f"Duration of the video file in seconds is inconsistent with the number of frames"
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
                    f"Video frame index is not an integer ({float(frame_idx)}), check your video file"
                )
            frame_idx = round(frame_idx)

            if frame_idx > next_frame_idx:
                raise ValueError(f"Video file is missing frame {next_frame_idx}")

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
    Provides sequential and random access to video frames. The frames are returned as NumPy arrays.

    Example usage:

    ```python
    with AVVideo(path) as video:
        print(len(video))               # total number of frames
        print(frame.shape)              # e.g. (1080, 1920, 3)
        print(frame.dtype)              # e.g. np.uint8
        frame = video[0]                # first frame
        frame = video[12]               # frame 12
        frame = video[len(video) - 1]   # last frame

        for frame in video:             # sequential iteration
            pass
    ```

    Videos with variable frame rates or inconsistent timing metadata may raise errors. This is
    intentional so such cases can be inspected and future support evaluated.

    Sequential access is generally faster than random access because random access may
    require seeking and decoding intermediate frames that are ultimately discarded.

    Thread type "AUTO" is generally faster for sequential access, but for random access it may be worse.
    """

    def __init__(
        self,
        path,
        format="rgb24",
        width=None,
        height=None,
        thread_type="SLICE",
        thread_count=0,
    ):
        """
        Initialize Video

        :param path: path to video file
        :param format: format when converting to numpy array (default rgb24, which is 8 bits per channel)
        see https://pyav.basswood-io.com/docs/stable/api/video.html#av.video.format.VideoFormat
        :param width: output width (None for same as video)
        :param height: output height (None for same as video)
        :param thread_type: thread type argument to pyav stream, must be 'SLICE' or 'FRAME', or 'AUTO'
        :param thread_count: thread count argument to pyav stream
        """

        self._av_video = AVVideo(
            path, thread_type=thread_type, thread_count=thread_count
        )
        self._format = format
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

    def __getitem__(self, item) -> np.ndarray:
        frame = self._av_video[item]
        return frame.to_ndarray(
            format=self._format, width=self._width, height=self._height
        )
