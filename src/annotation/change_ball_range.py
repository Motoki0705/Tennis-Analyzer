import json


def update_ball_bboxes(
    json_input_path: str,
    json_output_path: str,
    new_width: float,
    new_height: float,
    inplace: bool = False,
):
    """
    ボールのアノテーションにおける bbox をキーポイント中心に基づいて指定サイズに更新する関数。

    Parameters
    ----------
    json_input_path : str
        入力JSONファイルのパス。
    json_output_path : str
        出力JSONファイルのパス。
    new_width : float
        新しいバウンディングボックスの幅（ピクセル単位）。
    new_height : float
        新しいバウンディングボックスの高さ（ピクセル単位）。
    inplace : bool, default=False
        True の場合、入力と同じファイルに上書き保存します。
    """
    with open(json_input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated_count = 0
    for ann in data["annotations"]:
        if ann["category_id"] == 1:  # ボール
            keypoints = ann["keypoints"]
            x_center, y_center = keypoints[0], keypoints[1]

            # bboxの更新
            x_min = x_center - new_width / 2
            y_min = y_center - new_height / 2
            ann["bbox"] = [x_min, y_min, new_width, new_height]
            ann["area"] = new_width * new_height
            updated_count += 1

    # 保存
    output_path = json_input_path if inplace else json_output_path
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"{updated_count} 件のボールアノテーションを更新しました。")


update_ball_bboxes(
    json_input_path=r"data\ball\coco_annotations_all_merged.json",
    json_output_path=r"data\ball\coco_annotations_ball_ranged.json",
    new_height=8,
    new_width=8,
)
