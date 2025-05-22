import os
import shutil
from typing import List


def collect_files_by_extension(
    src_dirs: List[str],
    dst_dir: str,
    extensions: List[str],
    exclude_filenames: List[str] = None,
):
    """
    特定の拡張子を持ち、特定のファイル名を除外しつつ、複数ディレクトリからファイルを収集して1つのフォルダに保存する。

    Parameters:
        src_dirs (List[str]): ソースディレクトリのリスト
        dst_dir (str): 出力先ディレクトリ
        extensions (List[str]): 対象とする拡張子（例: ['.py', '.txt']）
        exclude_filenames (List[str]): 除外するファイル名（例: ['__init__.py']）
    """
    os.makedirs(dst_dir, exist_ok=True)
    extensions = [ext.lower() for ext in extensions]
    exclude_filenames = set(exclude_filenames or [])

    for src_root in src_dirs:
        for root, dirs, files in os.walk(src_root):
            for file in files:
                if file in exclude_filenames:
                    continue  # 除外リストにあればスキップ

                if os.path.splitext(file)[1].lower() in extensions:
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(dst_dir, file)

                    # 同名ファイルのリネーム処理
                    if os.path.exists(dst_path):
                        base, ext = os.path.splitext(file)
                        i = 1
                        while os.path.exists(dst_path):
                            dst_path = os.path.join(dst_dir, f"{base}_{i}{ext}")
                            i += 1

                    shutil.copy(src_path, dst_path)


collect_files_by_extension(
    src_dirs=["TrackNet"],  # 収集対象のフォルダ
    dst_dir="ballgit",  # 保存先
    extensions=[".py", ".yaml"],  # 対象拡張子
    exclude_filenames=["__init__.py"],
)
