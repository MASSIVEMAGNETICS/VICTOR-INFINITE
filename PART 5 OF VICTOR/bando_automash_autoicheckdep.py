# =================================================================
# FILE: video_quickslice_mashup.py
# VERSION: v1.0.0-BANDOCORE
# NAME: QuickfireVideoSlicer
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Splice N videos together, alternating every X seconds.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# =================================================================

from moviepy.editor import VideoFileClip, concatenate_videoclips
import sys

def quickfire_video_slicer(video_paths, slice_duration=2, output_path="output_bando_mashup.mp4"):
    # Load all videos
    clips = [VideoFileClip(v) for v in video_paths]
    min_duration = min([clip.duration for clip in clips])
    total_duration = int(min_duration // slice_duration) * slice_duration

    timeline = []
    time = 0
    idx = 0
    while time < total_duration:
        # Cycle through all videos, grabbing the next slice
        clip = clips[idx % len(clips)]
        start = time
        end = min(time + slice_duration, clip.duration)
        timeline.append(clip.subclip(start, end))
        time += slice_duration
        idx += 1

    # Concatenate all slices
    final = concatenate_videoclips(timeline, method="compose")
    final.write_videofile(output_path, codec="libx264", audio_codec="aac")
    print(f"Done! Output saved to {output_path}")

if __name__ == "__main__":
    # Example usage: python video_quickslice_mashup.py 2 vid1.mp4 vid2.mp4 vid3.mp4
    if len(sys.argv) < 4:
        print("USAGE: python video_quickslice_mashup.py [slice_seconds] [video1] [video2] ...")
        sys.exit(1)
    slice_sec = int(sys.argv[1])
    video_list = sys.argv[2:]
    quickfire_video_slicer(video_list, slice_duration=slice_sec)
