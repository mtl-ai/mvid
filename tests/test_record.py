from tempfile import NamedTemporaryFile

import av
import numpy as np

from mvid import Recorder


def test_record():

    with NamedTemporaryFile(prefix="test", suffix=".mp4", delete_on_close=False) as f:
        file_name = f.name

        width = 192
        height = 108
        fps = 24

        # add crf="0" option for lossless so that we can compare encoding with original frames
        with Recorder(file_name, fps=fps, codec_options=dict(crf="0")) as rec:
            for i in range(100):
                image = np.zeros((height, width, 3), dtype=np.uint8)
                image[i % height, i:] = 255
                rec(image)

        with av.open(file_name) as container:
            assert container.streams.video[0].base_rate == fps

            for i, frame in enumerate(container.decode(video=0)):
                frame = frame.to_ndarray(format="rgb24")

                assert frame.shape == (height, width, 3)

                target = np.zeros((height, width, 3), dtype=np.uint8)
                target[i % height, i:] = 255

                assert np.all(target == frame)

        # print(file_name)
        # input("paused")


if __name__ == "__main__":
    test_record()
