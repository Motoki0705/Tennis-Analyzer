import os
from pathlib import Path

# Gドライブのパスを指定（環境によってドライブ文字が異なる場合があります）
drive_path_str = "G:\\マイドライブ" # バックスラッシュは2つ重ねるか、'/'を使います
drive_path_obj = Path("G:/マイドライブ")

# --- ファイルの一覧を取得 ---
try:
    print(f"--- {drive_path_str} の中身 ---")
    for item in os.listdir(drive_path_str):
        print(item)
    print("-" * 20)

    # --- 新しいフォルダを作成 ---
    new_folder_path = drive_path_obj / "Python_Test_Folder"
    os.makedirs(new_folder_path, exist_ok=True)
    print(f"フォルダを作成しました: {new_folder_path}")

    # --- ファイルを書き込み ---
    file_path = new_folder_path / "my_local_file.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("これはローカルPCから直接書き込んだファイルです。")
    print(f"ファイルを書き込みました: {file_path}")

except FileNotFoundError:
    print(f"エラー: パス '{drive_path_str}' が見つかりません。")
    print("Google Drive for Desktopが実行されているか、ドライブ文字が正しいか確認してください。")
except Exception as e:
    print(f"エラーが発生しました: {e}")