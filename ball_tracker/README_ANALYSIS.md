# Ball Tracker Analysis Tools

ball_trackerの性能分析・可視化ツール群

## 🎯 目的

このツールを使って動画にball_trackerを適用し、以下を分析できます：

- **確信度分布の評価** - どの閾値が適切か判断
- **軌跡品質の確認** - 検出の一貫性と滑らかさ評価  
- **蒸留学習戦略の決定** - video_swin_transformerの学習計画策定

## 📁 ファイル構成

```
ball_tracker/
├── analysis_tool.py      # メイン分析クラス
├── batch_analysis.py     # バッチ処理ツール
├── run_analysis.py       # 簡易実行スクリプト
└── README_ANALYSIS.md    # このファイル
```

## 🚀 使用方法

### 1. 基本的な単一動画分析

```bash
# 簡易実行（推奨）
python run_analysis.py \
  --video /path/to/tennis_video.mp4 \
  --model_path /path/to/ball_tracker_model.pth.tar

# 詳細設定での実行
python analysis_tool.py \
  --video /path/to/tennis_video.mp4 \
  --model_path /path/to/ball_tracker_model.pth.tar \
  --output_dir ./analysis_results \
  --thresholds 0.3 0.5 0.7 0.8 0.9 \
  --device cuda
```

### 2. 複数動画の一括分析

```bash
python batch_analysis.py \
  --video_dir /path/to/tennis_videos/ \
  --model_path /path/to/ball_tracker_model.pth.tar \
  --output_dir ./batch_results \
  --max_workers 2
```

## 📊 分析結果の見方

### 出力ファイル

実行後、以下のファイルが生成されます：

```
analysis_results/
├── analysis_results.json      # 詳細な数値データ
├── statistics_summary.json    # 統計サマリー
├── analysis_overview.png      # 総合分析グラフ
└── threshold_analysis.png     # 閾値別詳細分析
```

### 重要な指標

**1. 可視率 (Visibility Ratio)**
- ボールが追跡できたフレームの割合
- 高いほど良好（目安: >70%）

**2. 平均確信度 (Average Confidence)**  
- 検出結果の信頼性
- 高いほど良好（目安: >0.6）

**3. 軌跡滑らかさ (Trajectory Smoothness)**
- 軌跡の一貫性（速度変化の標準偏差）
- 低いほど滑らか（目安: <50）

**4. 有効フレーム率 (Valid Frame Ratio)**
- 指定確信度閾値を超えるフレームの割合
- 蒸留学習で利用可能なデータ量を示す

## 🎯 蒸留学習戦略の決定

分析結果から以下の戦略を決定：

### 高品質データ生成 (推奨閾値: 0.8以上)
```
目的: 高精度なTeacherラベル生成
条件: 有効フレーム率 > 30%
効果: 少量だが高品質な学習データ
```

### バランス型 (推奨閾値: 0.6-0.7)
```
目的: 品質と量のバランス
条件: 有効フレーム率 > 50%
効果: 実用的な学習データ量を確保
```

### 大規模データ生成 (推奨閾値: 0.5以上)
```
目的: 大量データでの学習
条件: 有効フレーム率 > 70%
効果: 汎化性能向上、ただしノイズ含む
```

## 📈 分析結果の例

### コンソール出力例
```
📊 分析結果サマリー
============================================================
動画: tennis_match_001.mp4
総フレーム数: 1,500
可視フレーム数: 1,200 (80.0%)
平均検出数/フレーム: 2.3
平均確信度: 0.724
軌跡滑らかさ: 23.4

📈 確信度閾値別統計:
----------------------------------------
閾値  0.3: 有効率  95.2%, 平均確信度 0.651
閾値  0.5: 有効率  87.4%, 平均確信度 0.712
閾値  0.7: 有効率  65.1%, 平均確信度 0.798
閾値  0.8: 有効率  42.3%, 平均確信度 0.856
閾値  0.9: 有効率  18.7%, 平均確信度 0.924

🎯 推奨設定:
----------------------------------------
高品質用 (閾値0.8): 有効率 42.3%
バランス型 (閾値0.7): 有効率 65.1%  
大規模用 (閾値0.5): 有効率 87.4%
```

### 可視化グラフの見方

**analysis_overview.png** に含まれる6つのグラフ：

1. **確信度分布** - 検出結果の確信度ヒストグラム
2. **フレーム毎検出数** - 時系列での検出数変化
3. **ボール軌跡** - 2D空間でのボール移動経路
4. **閾値別有効率** - 各閾値での利用可能データ量
5. **時系列確信度** - フレーム毎の最大確信度推移
6. **統計サマリー** - 主要指標の数値一覧

## 🔧 トラブルシューティング

### よくある問題

**1. モデルが見つからない**
```bash
❌ モデルファイルが見つかりません
→ ball_trackerの.pth.tarファイルを正しいパスに配置
```

**2. CUDA メモリ不足** 
```bash
python run_analysis.py --video sample.mp4 --model_path model.pth.tar --device cpu
```

**3. 動画が読み込めない**
```bash
# 対応形式: .mp4, .avi, .mov, .mkv, .wmv, .flv
# ffmpegで変換: ffmpeg -i input.xxx -c copy output.mp4
```

**4. 依存関係エラー**
```bash
pip install matplotlib seaborn tqdm pandas opencv-python torch
```

### デバッグモード

詳細なログを確認したい場合：

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 分析実行
analyzer = BallTrackerAnalyzer(model_path, device)
results = analyzer.analyze_video(video_path, output_dir)
```

## 🚀 次のステップ

分析結果を基に以下に進んでください：

1. **適切な確信度閾値の決定**
2. **蒸留学習用データセットの生成** 
3. **video_swin_transformer の学習実行**

分析結果が良好（可視率>70%, 平均確信度>0.6）であれば、蒸留学習に進む準備完了です！

## 📚 参考情報

- ball_trackerの元実装: [WASB-SBDT](https://github.com/starashima/WASB-SBDT_sandbox)
- video_swin_transformer: `src/ball/models/video_swin_transformer.py`
- 蒸留学習実装: 分析結果を基に次回実装予定

---

**作成日**: 2024年  
**用途**: video_swin_transformer 蒸留学習のための事前分析 