# テスト実行スクリプト

このディレクトリには、各モジュールのテストを実行するためのスクリプトが含まれています。

## イベント検出モジュールのテスト

イベント検出モジュールのテストを実行するためのスクリプトが用意されています。

### 使用方法

#### Linuxまたは macOS:

```bash
# 実行権限を付与
chmod +x scripts/test/test_event_module.sh

# 実行
./scripts/test/test_event_module.sh
```

#### Windows (PowerShell):

```powershell
# 実行
powershell.exe -ExecutionPolicy Bypass -File scripts/test/test_event_module.ps1
```

#### Windows (コマンドプロンプト):

```cmd
scripts\test\test_event_module.bat
```

### 機能

- データセットのテスト実行 (`tests/data/test_event_dataset_io.py`)
- RNNモデルのテスト実行 (`tests/train_model_instantiate/test_event_models.py`)
- Transformerモデルのテスト実行 (`tests/train_model_instantiate/test_transformer_model.py`)
- テスト結果のログファイル保存 (`tests/logs/`)
- エラーが発生した場合の詳細表示
- テストの成功/失敗状態に基づく終了コード（成功: 0, 失敗: 1）

### ログファイル

テスト実行の詳細ログは以下の場所に保存されます：

- データセットテストログ: `tests/logs/event_dataset_test.log`
- RNNモデルテストログ: `tests/logs/event_model_test.log`
- Transformerモデルテストログ: `tests/logs/transformer_model_test.log`

エラーが発生した場合は、これらのログファイルを確認してください。 