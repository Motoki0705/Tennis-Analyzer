import cv2
import torch
import numpy as np
import albumentations as A
import logging
from typing import List, Tuple, Union
from pathlib import Path
from tqdm import tqdm
from scipy.special import expit
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

from src.court.models.fpn import CourtDetectorFPN
from src.utils.load_model import load_model_weights


ROW_LABELS = [
    ("top_left_doubles_corner",
     "top_left_singles_corner",
     "top_right_singles_corner",
     "top_right_doubles_corner"),            # 4
    ("top_left_service_corner",
     "top_service_T",
     "top_right_service_corner"),            # 3
    ("net_center",),                         # 1
    ("bottom_left_service_corner",
     "bottom_service_T",
     "bottom_right_service_corner"),         # 3
    ("bottom_left_doubles_corner",
     "bottom_left_singles_corner",
     "bottom_right_singles_corner",
     "bottom_right_doubles_corner"),         # 4
]
ROW_COUNTS = [len(r) for r in ROW_LABELS]     # [4,3,1,3,4]


class CourtPredictor:
    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "cpu",
        input_size: Tuple[int, int] = (256, 256),
        num_keypoints: int = 1,
        threshold: float = 0.5,
        min_distance: int = 10,
        radius: int = 5,
        kp_color: Tuple[int, int, int] = (0, 255, 0),
        use_half: bool = False
    ):
        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        self.device = device
        self.input_size = input_size
        self.threshold = threshold
        self.min_distance = min_distance
        self.radius = radius
        self.kp_color = kp_color
        self.use_half = use_half
        
        # モデルロード
        self.model = self._load_model(model_path, num_keypoints)

        # 変換
        self.transform = A.Compose([
            A.Resize(height=input_size[0], width=input_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            A.pytorch.ToTensorV2()
        ])

    def _load_model(self, model_path: str, num_keypoints: int) -> torch.nn.Module:
        self.logger.info(f"Loading model with num_keypoints={num_keypoints} from {model_path}")
        model = CourtDetectorFPN()
        model = load_model_weights(model, model_path)
        model = model.eval().to(self.device)
        return model

    def predict(self, frames: List[np.ndarray]) -> List[List[dict]]:
        """
        入力フレーム群に対してコートキーポイント推論を行う。
        returns: List of List of {"x": int, "y": int, "confidence": float}
        """
        tensors = []
        for img in frames:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            aug = self.transform(image=rgb)
            tensors.append(aug["image"])
        batch = torch.stack(tensors).to(self.device)  # (B, C, H, W)

        if self.use_half:
            with torch.no_grad(), torch.amp.autocast(device_type=self.device, dtype=torch.float16):
                outputs = self.model(batch)
        else:
            with torch.no_grad():
                outputs = self.model(batch)

        if outputs.ndim == 4 and outputs.shape[1] == 1:
            heatmaps = outputs[:, 0]
        else:
            heatmaps = outputs.sum(dim=1)

        heatmaps = expit(heatmaps.cpu().numpy())  # シグモイド正規化

        results = []
        for hm, frame in zip(heatmaps, frames):
            keypoints = self._extract_keypoints(hm, frame.shape[:2])
            results.append(keypoints)

        return results

    def _extract_keypoints(self, heatmap: np.ndarray, orig_shape: Tuple[int, int]) -> List[dict]:
        """
        シグモイド後のヒートマップからキーポイント座標を抽出する。
        returns: [{"x": int, "y": int, "confidence": float}, ...]
        """
        h_heat, w_heat = heatmap.shape
        H, W = orig_shape

        # 局所最大検出
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(heatmap, kernel)
        peaks_mask = (heatmap == dilated) & (heatmap > self.threshold)
        ys, xs = np.where(peaks_mask)

        # NMS
        scores = heatmap[ys, xs]
        order = np.argsort(scores)[::-1]
        selected = []
        for idx in order:
            y, x = int(ys[idx]), int(xs[idx])
            if all(abs(y - yy) + abs(x - xx) > self.min_distance for yy, xx in selected):
                selected.append((y, x))

        keypoints = []
        for y, x in selected:
            X = int(x * W / w_heat)
            Y = int(y * H / h_heat)
            confidence = float(heatmap[y, x])
            keypoints.append({
                "x": X,
                "y": Y,
                "confidence": confidence
            })

        return keypoints
    
    @staticmethod
    def assign_labels_robust(raw_kps, img_h, img_w):
        """
        raw_kps: List[dict]  - {"x","y","confidence"}
        return: OrderedDict[label] = {"x","y","score","missing"}
        """
        # ----- 前処理 -----
        pts   = np.array([[kp["x"], kp["y"]] for kp in raw_kps])
        scr   = np.array([kp.get("confidence", 1.0) for kp in raw_kps])
        m, k  = pts.shape[0], 5
        k     = min(k, m)                          # 点が少ない場合でもクラスタ可能

        # ----- Row クラスタリング (y) -----
        km    = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(pts[:,1][:,None])
        row_id= km.labels_
        centers_y = km.cluster_centers_.ravel()
        rows_by_y= np.argsort(centers_y)           # 小さい y → Row0

        # map: 行番号(0~4) -> 点 index list
        rows = {r: [] for r in range(5)}
        for idx, cid in enumerate(row_id):
            # cid が rows_by_y に何番目で現れるか = 行番号推定
            r = np.where(rows_by_y == cid)[0][0]
            rows[r].append(idx)

        # ----- 行欠損補完 (y 座標) -----
        # 無い行は線形補間 / 外挿で y を生成
        valid_rows = [r for r,v in rows.items() if v]
        y_map = {r: pts[rows[r],1].mean() for r in valid_rows}
        for r in range(5):
            if r not in y_map:
                # 内挿 or 外挿
                lower = max([rv for rv in valid_rows if rv<r], default=None)
                upper = min([rv for rv in valid_rows if rv>r], default=None)
                if lower is not None and upper is not None:
                    # 線形補間
                    y_map[r] = np.interp(r, [lower, upper], [y_map[lower], y_map[upper]])
                elif lower is not None:
                    # 外挿(下側)
                    y_map[r] = y_map[lower] + (r-lower)*(y_map[lower]-y_map[lower-1])
                elif upper is not None:
                    # 外挿(上側)
                    y_map[r] = y_map[upper] - (upper-r)*(y_map[upper+1]-y_map[upper])

        # ----- 各行ごとに x ソート → 列不足補完 -----
        labeled = {}
        for r in range(5):
            idxs = rows.get(r, [])
            row_pts = pts[idxs] if idxs else np.empty((0,2))
            row_scr = scr[idxs] if idxs.size else np.empty(0)

            # 既存点を x 昇順でソート
            order = np.argsort(row_pts[:,0]) if idxs else []
            expect = ROW_COUNTS[r]
            xs, ys, scs = [], [], []

            for o in order:
                xs.append(float(row_pts[o,0]))
                ys.append(float(row_pts[o,1]))
                scs.append(float(row_scr[o]))

            # 列欠損補完：両端があれば線形内挿，片端なら対称幅推定
            if len(xs) >= 2:
                # 幅
                full_range = (min(xs), max(xs))
                full_span  = full_range[1]-full_range[0]
                # 理想比で内挿分配
                missing = expect - len(xs)
                if missing>0:
                    step = full_span/(expect-1)
                    # 目標位置列を作成
                    target_xs = [full_range[0] + i*step for i in range(expect)]
                    filled=[]
                    for tx in target_xs:
                        if all(abs(tx - x) > step*0.3 for x in xs):
                            filled.append(tx)
                    for tx in filled[:missing]:
                        xs.append(tx); ys.append(y_map[r]); scs.append(0.0)
            else:
                # xs が0または1の場合：行幅を近隣行からコピーして片側外挿
                ref = 0 if r<2 else 4            # 上側欠損→Row0, 下側→Row4
                ref_span = (max(pts[:,0])-min(pts[:,0]))*0.9
                base_x   = xs[0] if xs else (img_w*0.5)
                step     = ref_span/(expect-1)
                xs = [base_x + (i-int(expect/2))*step for i in range(expect)]
                ys = [y_map[r]]*expect
                scs= [0.0]*expect

            # 最終並べ替え
            sorted_pack = sorted(zip(xs, ys, scs), key=lambda x: x[0])
            for col, (xx,yy,ss) in enumerate(sorted_pack):
                label = ROW_LABELS[r][col]
                labeled[label] = {"x": xx, "y": yy,
                                "confidence": ss, "missing": ss==0.0}

        return labeled

    def overlay(self, frame: np.ndarray, keypoints: List[dict]) -> np.ndarray:
        """
        入力フレームとキーポイント群を受け取り、描画して返す。
        """
        for kp in keypoints:
            if kp["confidence"] >= self.threshold:
                cv2.circle(
                    frame,
                    (kp["x"], kp["y"]),
                    self.radius,
                    self.kp_color,
                    thickness=-1,
                    lineType=cv2.LINE_AA
                )
        return frame
    
    def label_points(self, kps: List[dict], shape: Tuple[int,int]):
        H, W = shape
        return self.assign_labels_robust(kps, H, W)

    def overlay_skeleton(self, frame: np.ndarray, labeled_pts: dict):
        SKELETON_EDGES = [
            # 外枠
            ("top_left_doubles_corner", "top_right_doubles_corner"),
            ("top_right_doubles_corner", "bottom_right_doubles_corner"),
            ("bottom_right_doubles_corner", "bottom_left_doubles_corner"),
            ("bottom_left_doubles_corner", "top_left_doubles_corner"),
            # シングルス
            ("top_left_singles_corner", "bottom_left_singles_corner"),
            ("top_right_singles_corner", "bottom_right_singles_corner"),
            # サービス
            ("top_left_service_corner","top_right_service_corner"),
            ("bottom_left_service_corner","bottom_right_service_corner"),
            # センター
            ("top_service_T","bottom_service_T"),
        ]
        # 点
        for d in labeled_pts.values():
            cv2.circle(frame,(int(d["x"]),int(d["y"])),4,
                       (0,255,0) if not d["missing"] else (0,128,255),-1,cv2.LINE_AA)
        # 線
        for a,b in SKELETON_EDGES:
            pa, pb = labeled_pts[a], labeled_pts[b]
            cv2.line(frame, (int(pa["x"]),int(pa["y"])),
                            (int(pb["x"]),int(pb["y"])),
                            (255,255,0),1,cv2.LINE_AA)
        return frame
    

    def run(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        batch_size: int = 8
    ) -> None:
        """
        動画を読み込み、batch_size フレームずつまとめて推論→オーバーレイし、
        出力動画に書き出します。
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            self.logger.error(f"動画ファイルを開けませんでした: {input_path}")
            return

        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        self.logger.info(
            f"読み込み完了 → フレーム数: {total}, FPS: {fps:.2f}, 解像度: {width}×{height}"
        )

        batch: List[np.ndarray] = []
        with tqdm(total=total, desc="Court 推論処理") as pbar:
            # フレームの読み込み＋バッチ推論ループ
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                batch.append(frame)
                # バッチがたまったらまとめて推論
                if len(batch) == batch_size:
                    kps_batch = self.predict(batch)
                    for frm, raw_kps in zip(batch, kps_batch):
                        labeled = self.label_points(raw_kps, frm.shape[:2])
                        overlaid = self.overlay_skeleton(frm, labeled)
                        writer.write(overlaid)
                        pbar.update(1)
                    batch.clear()

            # 残りフレームの処理
            if batch:
                kps_batch = self.predict(batch)
                for frm, raw_kps in zip(batch, kps_batch):
                    labeled = self.label_points(raw_kps, frm.shape[:2])
                    overlaid = self.overlay_skeleton(frm, labeled)
                    writer.write(overlaid)
                    pbar.update(1)

        cap.release()
        writer.release()
        self.logger.info(f"処理完了 → 出力ファイル: {output_path}")
