"""
ボール検出用LightningModuleパッケージ
"""
from src.ball.lit_module.heatmap_regression_lit_module import HeatmapRegressionLitModule
from src.ball.lit_module.coord_regression_lit_module import CoordRegressionLitModule

# エイリアスを設定（設定ファイルとの互換性のため）
CatFramesLitModule = HeatmapRegressionLitModule 