#!/usr/bin/env python
# coding: utf-8

import torch
import argparse
import os

def show_checkpoint_keys(
    ckpt_path: str,
    show_shapes: bool = False,
    show_hparams: bool = False
):
    """
    チェックポイントファイル（.pt/.pth/.ckpt）から
    - state_dict のキー一覧
    - （オプション）各パラメータの shape
    - （オプション）保存されたハイパーパラメータ
    を表示します。
    """
    # ファイル存在チェック
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"ファイルが見つかりません: {ckpt_path}")

    # CPU 上でロード
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # ── ハイパーパラメータ表示 ──
    if show_hparams:
        # Lightning の古いバージョンでは "hyper_parameters"、
        # 新しいバージョンでは "hparams" キーで保存されている場合があります
        hps = None
        if isinstance(ckpt, dict):
            hps = ckpt.get("hyper_parameters") or ckpt.get("hparams")
        if isinstance(hps, dict) and hps:
            print("===== ハイパーパラメータ =====")
            for k, v in hps.items():
                print(f"{k}: {v}")
            print()
        else:
            print("チェックポイントにハイパーパラメータは見つかりませんでした。\n")

    # ── state_dict の取得 ──
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state_dict = ckpt
    else:
        raise ValueError(f"予期しないチェックポイント形式です: {ckpt_path}")

    # ── キー一覧表示 ──
    print(f"===== チェックポイント: {ckpt_path} =====\n")
    for key, tensor in state_dict.items():
        if show_shapes and isinstance(tensor, torch.Tensor):
            print(f"{key}\t→ shape: {tuple(tensor.shape)}")
        else:
            print(key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch(.pt/.pth)／Lightning(.ckpt)チェックポイントの state_dict キー一覧を表示します。"
    )
    parser.add_argument(
        "--ckpt_path", "-c",
        required=True,
        type=str,
        help="チェックポイントファイルのパス (.pt/.pth/.ckpt)"
    )
    parser.add_argument(
        "--shapes", "-s",
        action="store_true",
        help="各パラメータの shape も表示する"
    )
    parser.add_argument(
        "--hparams", "-p",
        action="store_true",
        help="保存されたハイパーパラメータも表示する"
    )
    args = parser.parse_args()

    try:
        show_checkpoint_keys(
            args.ckpt_path,
            show_shapes=args.shapes,
            show_hparams=args.hparams
        )
    except Exception as e:
        print(f"エラー: {e}")
