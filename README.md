# mvid
`mvid` is a simple library to access video frames by index. This is useful when
you want to refer to a specific frame in a video, but you prefer not dealing with timestamps, keyframes,
and other video encoding details. 

It is implemented as a very light interface ontop of [PyAV](https://pyav.basswood-io.com/docs/stable/).

# Usage
To read from a video, use the `Video` class. Frames are returned as NumPy arrays.

```python
import mvid

with mvid.Video("myvideo.mp4") as video:
    # get the number of frames
    print(len(video))

    # random access
    frame = video[57]  # frame is a NumPy array
    
    print(frame.shape)  # (H, W, 3)
    
    # iterate over all frames in the video
    for frame in video:
        pass
```

We also give a simple `Recorder` class to output to a video file.
```python
import numpy as np
import mvid

with mvid.Recorder("output.mp4", fps=50) as rec:
    
    # record 1 second of gray
    for _ in range(50):
        rec(128 * np.ones(1080, 1920, 3))

```

# Installation
```bash
pip install mvid
```

It requires [PyAV](https://pyav.basswood-io.com/docs/stable/) and NumPY.

# How it works
Frame lookup is based on decoding from the nearest preceding keyframe up to the requested index. 
We determine that index using each frame’s timestamp together with the stream’s frame rate. 
This approach works well for videos with consistent timing metadata, but not all files follow those assumptions. 
Some containers use variable frame rates or contain incomplete or inconsistent timestamps. In those cases 
there is no reliable way to infer a frame index without first scanning every frame and assigning 
indices explicitly. Rather than performing that preprocessing step, we intentionally crash when encountering 
timing metadata that cannot be interpreted unambiguously.

# Performance
Generally speaking, sequential access is as fast as possible thanks to PyAV. Check `benchmark.py` and compare
with `ffmpeg -i <my_video> -f null -`. The benchmarking script will also try random access and various 
thread parameters so you can see what performance to expect. 

There is overhead from conversion to NumPy arrays (which is calling `av.VideoFrame.to_ndarray()`). 
We also provide a "raw" AVVideo class that performs all the bookkeeping for frame access without NumPy conversion.

# Related projects
[TorchCodec](https://github.com/meta-pytorch/torchcodec) is a more heavy-duty library that returns PyTorch tensors.
It also has index-based access (among other options). It requires managing your installation of ffmpeg.