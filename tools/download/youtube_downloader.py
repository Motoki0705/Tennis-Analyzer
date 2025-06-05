import yt_dlp

def download_video(url, output_dir="downloads", format="best"):
    ydl_opts = {
        'outtmpl': f'{output_dir}/%(title)s.%(ext)s',  # 保存ファイル名のテンプレート
        'format': format,  # ダウンロードするフォーマット（best, mp4 など）
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

if __name__ == "__main__":
    # ダウンロード対象の動画URL
    url = "https://www.youtube.com/watch?v=twpi3NIk2wQ&ab_channel=RogerThatTennis"
    download_video(url, output_dir="data/videos")
