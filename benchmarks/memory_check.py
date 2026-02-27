import random
import sys

from tqdm import tqdm

from mvid import Video

if __name__ == "__main__":
    # In an older implementation there was an issue with resources not being freed properly.
    # The cause was circular references with the frame generator.
    # This was especially obvious with "AUTO" threading.
    video_path = sys.argv[1]
    window_size = 5

    tqdm.write("Observe system memory usage and check for any leaks")

    pbar = tqdm()
    while True:
        with Video(video_path, thread_type="AUTO") as video:
            frame_idx = random.randint(0, len(video) - 1)
            _ = [video[(frame_idx + i) % len(video)] for i in range(window_size)]
            pbar.update(1)
