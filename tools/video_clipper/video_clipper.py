import cv2
import os

def clip_video(input_path: str, output_path: str, start_time_sec: float = 0.0, end_time_sec: float | None = None, start_frame: int = 0, end_frame: int | None = None):
    """Clips a video file based on specified time or frame range.

    Args:
        input_path: Path to the input video file.
        output_path: Path to save the output clip file.
        start_time_sec: Start time of the clip in seconds. Defaults to 0.0.
        end_time_sec: End time of the clip in seconds. If None, clips until the end.
        start_frame: Start frame of the clip. Defaults to 0.
        end_frame: End frame of the clip. If None, clips until the end.
    """
    # Add error handling for file not found, invalid paths, etc.
    if not os.path.exists(input_path):
        print(f"Error: Input video file not found at {input_path}")
        return

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print(f"Error: Could not create output directory {output_dir}: {e}")
            return

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps

    # Determine start and end frames based on provided arguments
    # Prioritize time-based clipping if provided
    if end_time_sec is not None:
        start_frame_to_use = int(start_time_sec * fps)
        end_frame_to_use = int(end_time_sec * fps)
    elif end_frame is not None:
        start_frame_to_use = start_frame
        end_frame_to_use = end_frame
    else: # Clip until the end
        start_frame_to_use = start_frame
        end_frame_to_use = frame_count


    if start_frame_to_use < 0:
        start_frame_to_use = 0
    if end_frame_to_use > frame_count:
        end_frame_to_use = frame_count
    if start_frame_to_use >= end_frame_to_use:
        print("Warning: Start position is at or after end position. No clip will be generated.")
        cap.release()
        return


    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_to_use)

    # Get video properties for VideoWriter
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # TODO: Determine appropriate codec (e.g., 'mp4v', 'xvid') based on desired output format
    # For .mp4, 'mp4v' or 'avc1' (H.264) are common
    # For .avi, 'XVID' is common
    # Need to check available codecs on the system

    # Simple approach: use a common codec like mp4v for .mp4 output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Example codec for .mp4
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Add error handling for VideoWriter creation
    if not out.isOpened():
        print(f"Error: Could not create video writer for {output_path}. Check codec support and file path.")
        cap.release()
        return

    print(f"Clipping video from frame {start_frame_to_use} to {end_frame_to_use} (approx {start_frame_to_use/fps:.2f}s to {end_frame_to_use/fps:.2f}s)")

    current_frame = start_frame_to_use
    while cap.isOpened() and current_frame < end_frame_to_use:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        current_frame += 1

    cap.release()
    out.release()
    print(f"Clip saved to {output_path}")

if __name__ == '__main__':
    # Example usage (for testing purposes)
    # TODO: Replace with actual video file and desired output path
    # input_video = "path/to/your/video.mp4"
    # output_clip = "path/to/your/output_clip.mp4"
    # clip_video(input_video, output_clip, start_time_sec=5.0, end_time_sec=10.0)
    pass
