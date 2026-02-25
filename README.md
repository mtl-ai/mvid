# mvid
mvid is a simple library to treat video as a sequence (e.g. as a list) of NumPY arrays.

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
It is built on top of PyAV (with minimal to no overhead) and abstracts away seeking and timing issues. 