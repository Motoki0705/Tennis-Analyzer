import yt_dlp
import os
from pathlib import Path  # pathlibを使い、OSに依存しない安定したパス操作を実現

def get_downloaded_urls(record_file):
    """記録ファイルからダウンロード済みのURLリストを読み込む"""
    record_path = Path(record_file)
    if not record_path.exists():
        return set()
    # 文字コードを指定してファイルを開く
    with record_path.open('r', encoding='utf-8') as f:
        return {line.strip() for line in f if line.strip()}

def add_downloaded_url(record_file, url):
    """URLを記録ファイルに追加する"""
    with open(record_file, 'a', encoding='utf-8') as f:
        f.write(url + '\n')

def download_videos_renamed(urls, base_output_dir="data/videos", record_file="downloaded_urls.txt"):
    """
    動画リストをフルHD(1080p)でダウンロードし、連番で名前を付ける。
    ダウンロード済みのURLはスキップし、番号付けを再開する。
    """
    # Pathオブジェクトで出力ディレクトリを扱う
    base_dir_path = Path(base_output_dir)
    base_dir_path.mkdir(parents=True, exist_ok=True)

    already_downloaded = get_downloaded_urls(record_file)
    print(f"Already downloaded {len(already_downloaded)} URLs.")

    # 既存のファイルから最大のゲーム番号を取得
    max_game_num = 0
    for f in base_dir_path.iterdir():
        if f.is_file() and f.name.startswith('game'):
            try:
                # 'game1.mp4' -> 'game1' -> '1'
                num_part = f.stem.replace('game', '')
                max_game_num = max(max_game_num, int(num_part))
            except (ValueError, IndexError):
                pass # 'game'で始まるが数字が続かないファイルは無視

    next_game_num = max_game_num + 1
    download_counter = 0

    # ダウンロードが必要なURLだけをリストアップ
    urls_to_process = [url for url in urls if url not in already_downloaded]
    if not urls_to_process:
        print("No new videos to download.")
        return

    for url in urls_to_process:
        current_game_num = next_game_num + download_counter
        
        # ★修正点: yt-dlpに出力ファイル名を直接指定させる
        # これにより、ダウンロード後のファイル名変更が不要になる
        output_template = base_dir_path / f"game{current_game_num}.%(ext)s"

        ydl_opts = {
            'outtmpl': str(output_template),
            # AV1(av01)を避け、H.264(avc)を優先的に選択するよう変更
            'format': 'bestvideo[height<=1080][vcodec^=avc]+bestaudio/bestvideo[height<=1080]+bestaudio/best',
            'quiet': True,
            'no_warnings': True,
            'merge_output_format': 'mp4',
        }
        
        print(f"Downloading video from {url} as game{current_game_num}...")
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # extract_info(download=True)ではなく、download()を直接呼ぶ
                ydl.download([url])
            
            print(f"Finished downloading and saved as game{current_game_num}.")
            add_downloaded_url(record_file, url)
            download_counter += 1
        except yt_dlp.utils.DownloadError as e:
            # yt-dlp固有のダウンロードエラーをハンドリング
            print(f"Error downloading {url}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred with {url}: {e}")


if __name__ == "__main__":
    urls_to_download = [
        "https://www.youtube.com/watch?v=EPLKsPXz1TM&t=17s",
        "https://www.youtube.com/watch?v=xp2mYmNl-lg&t=210s"
    ]
    
    # スクリプトの実行場所からの相対パスとしてPathオブジェクトを使用
    base_dir = Path("data/raw/videos/")
    record_file_path = Path("tools/download/downloaded_urls.txt")
    
    download_videos_renamed(urls_to_download, base_output_dir=base_dir, record_file=record_file_path)