"""
ボール検出用LightningModuleパッケージ
"""
from src.ball.lit_module.heatmap_regression_lit_module import HeatmapRegressionLitModule
from src.ball.lit_module.coord_regression_lit_module import CoordRegressionLitModule
from src.ball.lit_module.self_training_lit_module import SelfTrainingLitModule, SelfTrainingCoordLitModule

# エイリアスを設定（設定ファイルとの互換性のため）
CatFramesLitModule = HeatmapRegressionLitModule 