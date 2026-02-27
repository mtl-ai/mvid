import av
import numpy as np
from tempfile import NamedTemporaryFile

from mvid import Video

from random import shuffle


def test_seek():

    with NamedTemporaryFile(prefix="test", suffix=".mp4", delete_on_close=False) as f:
        file_name = f.name

        # generate test video with keyframes every 10 frames
        with av.open(file_name, "w") as c:
            stream = c.add_stream("libx264", rate=50)
            stream.width = 640
            stream.height = 480
            # print(dir(stream.codec_context))
            stream.codec_context.gop_size = 11

            for i in range(50):
                image = np.random.randint(
                    0, 255, size=(stream.height, stream.width, 3), dtype=np.uint8
                )
                frame = av.VideoFrame.from_ndarray(image)
                for packet in stream.encode(frame):
                    c.mux(packet)

            # flush
            for packet in stream.encode():
                c.mux(packet)

        # print(file_name)
        # input("paused")

        # decode the encoded frames
        with av.open(file_name, "r") as c:
            target_frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]

        # compare sequential and random access
        with Video(file_name) as video:
            actual_frames = [f for f in video]
            assert len(actual_frames) == len(target_frames)
            for a, t in zip(actual_frames, target_frames):
                assert np.all(a == t)

            frame_indices = list(range(len(target_frames)))
            shuffle(frame_indices)
            # print(frame_indices)
            for idx in frame_indices:
                assert np.all(video[idx] == actual_frames[idx])


if __name__ == "__main__":
    test_seek()
