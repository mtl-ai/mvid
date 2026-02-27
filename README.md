# mvid
mvid is a simple library to access videos by frame index and return NumPy arrays.

```python
from mvid import Video

with Video("myvideo.mp4") as video:
    # get the number of frames
    print(len(video))

    # random access
    frame = video[57]
    
    # iterate over all frames in the video
    for frame in video:
        pass
```
It uses PyAV (with minimal to no overhead) and abstracts away seeking logic for you.

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
there is no reliable way to infer a stable frame index without first scanning every frame and assigning 
indices explicitly. Rather than performing that preprocessing step, we intentionally crash when encountering 
timing metadata that cannot be interpreted unambiguously.

# Performance
Generally speaking, sequential access is as fast as possible thanks to PyAV. Check `benchmark.py` and compare
with `ffmpeg -i <my_video> -f null -`. The benchmarking script will also try random access and various 
thread parameters so you can see what performance to expect. 

There is overhead from conversion to NumPY arrays. We also provide a more "raw" AVVideo class that 
performs all the bookkeeping without NumPY conversion.

# Related projects
[TorchCodec](https://github.com/meta-pytorch/torchcodec) is a more heavy-duty library that returns PyTorch tensors.
It also has index-based access (among other options). It requires managing your installation of ffmpeg.