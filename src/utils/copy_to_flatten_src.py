import os
import shutil

src_root = r'C:\Users\kamim\code\Tennis-Analyzer\TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch\src'
dst_folder = r'C:\Users\kamim\code\Tennis-Analyzer\TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch/all_code'
os.makedirs(dst_folder, exist_ok=True)

for root, dirs, files in os.walk(src_root):
    for file in files:
        if file.endswith('.py'):
            src_path = os.path.join(root, file)
            dst_path = os.path.join(dst_folder, file)

            # 同名ファイルがある場合はリネーム
            if os.path.exists(dst_path):
                base, ext = os.path.splitext(file)
                i = 1
                while os.path.exists(dst_path):
                    dst_path = os.path.join(dst_folder, f"{base}_{i}{ext}")
                    i += 1

            shutil.copy(src_path, dst_path)
