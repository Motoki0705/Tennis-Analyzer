# multi_flow_annotator/file_utils.py

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np

def validate_paths(input_dir: Path, output_json: Path) -> None:
    """入力ディレクトリと出力パスの有効性を検証します。

    Args:
        input_dir (Path): 検証する入力ディレクトリのパス。
        output_json (Path): 検証する出力JSONファイルのパス。

    Raises:
        ValueError: 入力ディレクトリが存在しないか、ディレクトリでない場合に発生します。
    """
    if not input_dir.is_dir():
        raise ValueError(f"入力パスはディレクトリである必要があります: {input_dir}")
    output_json.parent.mkdir(parents=True, exist_ok=True)


def extract_ids_from_path(img_path: Path) -> Tuple[Optional[int], Optional[int]]:
    """ファイルパスから 'game' と 'clip' のIDを抽出します。

    Args:
        img_path (Path): 分析する画像ファイルのパス。

    Returns:
        Tuple[Optional[int], Optional[int]]: (game_id, clip_id) のタプル。見つからない場合はNone。
    """
    game_id, clip_id = None, None
    for part in img_path.parts:
        part_lower = part.lower()
        if part_lower.startswith('game'):
            try: game_id = int(part_lower[4:])
            except ValueError: pass
        elif part_lower.startswith('clip'):
            try: clip_id = int(part_lower[4:])
            except ValueError: pass
    return game_id, clip_id


def extract_frame_number(path: Path) -> int:
    """ファイル名からフレーム番号を抽出します。

    Args:
        path (Path): フレーム番号を抽出するファイルのパス。

    Returns:
        int: 抽出されたフレーム番号。抽出できない場合は0。
    """
    digits = ''.join(filter(str.isdigit, path.stem))
    return int(digits) if digits else 0


def collect_and_group_images(
    input_dir: Path, extensions: List[str]
) -> Dict[Tuple[int, int], List[Path]]:
    """指定されたディレクトリから画像を収集し、game/clip IDでグループ化します。

    Args:
        input_dir (Path): 画像を検索するルートディレクトリ。
        extensions (List[str]): 収集する画像の拡張子リスト (例: ['.jpg', '.png'])。

    Returns:
        Dict[Tuple[int, int], List[Path]]: (game_id, clip_id) をキーとし、
                                           画像パスのリストを値とする辞書。
    """
    grouped_files = {}
    all_files = []
    for ext in extensions:
        all_files.extend(input_dir.glob(f"**/*{ext}"))

    for img_path in all_files:
        game_id, clip_id = extract_ids_from_path(img_path)
        group_key = (game_id or -1, clip_id or -1)
        if group_key not in grouped_files:
            grouped_files[group_key] = []
        grouped_files[group_key].append(img_path)

    # 各グループ内をフレーム番号でソート
    for key in grouped_files:
        grouped_files[key].sort(key=extract_frame_number)

    return grouped_files


def load_frame(img_path: Path) -> Optional[np.ndarray]:
    """ディスクから単一の画像フレームを読み込みます。

    Args:
        img_path (Path): 読み込む画像ファイルのパス。

    Returns:
        Optional[np.ndarray]: 読み込まれた画像データ(BGR)。失敗した場合はNone。
    """
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"警告: 画像の読み込みに失敗しました: {img_path}")
        return None
    return frame