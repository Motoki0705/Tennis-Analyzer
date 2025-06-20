import yt_dlp
import os

def get_downloaded_urls(record_file):
    """Reads the list of already downloaded URLs from a file."""
    downloaded_urls = set()
    if os.path.exists(record_file):
        with open(record_file, 'r') as f:
            for line in f:
                downloaded_urls.add(line.strip())
    return downloaded_urls

def add_downloaded_url(record_file, url):
    """Adds a URL to the record file."""
    with open(record_file, 'a') as f:
        f.write(url + '\n')

def download_videos_renamed(urls, base_output_dir="data/videos", format="best", record_file="downloaded_urls.txt"):
    """
    Downloads a list of videos, renaming them sequentially (e.g., game1.mp4, game2.mp4).
    Resumes numbering if files already exist and prevents duplicate downloads.

    Args:
        urls (list): A list of video URLs to download.
        base_output_dir (str): The directory where the renamed video files will be saved.
        format (str): The desired format for the downloaded videos (e.g., "best", "mp4").
        record_file (str): The file to store a record of downloaded URLs.
    """
    # Create the base output directory if it doesn't exist
    os.makedirs(base_output_dir, exist_ok=True)

    # Get already downloaded URLs
    already_downloaded = get_downloaded_urls(record_file)
    print(f"Already downloaded {len(already_downloaded)} URLs.")

    # Determine the starting game number based on existing files like game1.mp4, game2.mp4
    max_game_num = 0
    for filename in os.listdir(base_output_dir):
        if filename.startswith('game') and '.' in filename:
            try:
                # Extract the number before the first dot (for extension)
                num_part = filename.split('.')[0].replace('game', '')
                max_game_num = max(max_game_num, int(num_part))
            except ValueError:
                pass # Ignore files that don't match the numbering pattern

    next_game_num = max_game_num + 1

    for i, url in enumerate(urls):
        if url in already_downloaded:
            print(f"Skipping {url} - already downloaded.")
            continue

        # Use a temporary filename initially for yt-dlp to handle extension correctly
        # We'll rename it after download
        temp_filepath_template = os.path.join(base_output_dir, 'temp_download_%(ext)s')

        ydl_opts = {
            'outtmpl': temp_filepath_template,
            'format': format,
            'updatetime': False, # Avoid updating modification time to prevent issues with rename
            'quiet': True,      # Suppress verbose yt-dlp output
            'no_warnings': True # Suppress warnings
        }
        
        print(f"Downloading video from {url}...")
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Download the video; info_dict contains file path after download
                info_dict = ydl.extract_info(url, download=True)
                downloaded_filepath = ydl.prepare_filename(info_dict)

            # Determine the new desired filename
            file_extension = os.path.splitext(downloaded_filepath)[1] # Get .mp4, .webm etc.
            new_filename = f"game{next_game_num + i}{file_extension}"
            new_filepath = os.path.join(base_output_dir, new_filename)

            # Rename the downloaded file
            os.rename(downloaded_filepath, new_filepath)
            print(f"Finished downloading and renaming video to {new_filepath}.")
            
            # Add the URL to the record file after successful download and rename
            add_downloaded_url(record_file, url)
        except Exception as e:
            print(f"Error downloading {url}: {e}")


if __name__ == "__main__":
    urls_to_download = [
        "https://www.youtube.com/watch?v=dvBr9Wr8BCY",
        "https://www.youtube.com/watch?v=EPLKsPXz1TM",
        "https://www.youtube.com/watch?v=vR5ykbDSV-4",
        "https://www.youtube.com/watch?v=mK5qvAZaTSk",
        "https://www.youtube.com/watch?v=SOrzYmtUEzQ"
    ]
    
    base_dir = "tools/annotation/data/annotation_workspace/videos"
    record_file_path = "tools/download/downloaded_urls.txt" # You can specify a different path if needed
    
    download_videos_renamed(urls_to_download, base_output_dir=base_dir, record_file=record_file_path)