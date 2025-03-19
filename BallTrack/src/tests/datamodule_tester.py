class DataModuleTester:
    def __init__(self, datamodule):
        """
        Args:
            datamodule: LightningDataModuleのインスタンス
        """
        self.datamodule = datamodule

    def test(self):
        # fitステージのテスト
        print("=== 'fit' stage のテスト開始 ===")
        self.datamodule.setup(stage="fit")

        train_loader = self.datamodule.train_dataloader()
        print(f"Train DataLoader: バッチ数 = {len(train_loader)}")
        self._inspect_dataloader(train_loader, stage="train")

        val_loader = self.datamodule.val_dataloader()
        print(f"Validation DataLoader: バッチ数 = {len(val_loader)}")
        self._inspect_dataloader(val_loader, stage="val")

        # testステージのテスト
        print("=== 'test' stage のテスト開始 ===")
        self.datamodule.setup(stage="test")
        test_loader = self.datamodule.test_dataloader()
        print(f"Test DataLoader: バッチ数 = {len(test_loader)}")
        self._inspect_dataloader(test_loader, stage="test")

        print("DataModuleのテストが正常に完了しました。")

    def _inspect_dataloader(self, dataloader, stage=""):
        """
        dataloaderから1バッチ分取り出し、データの形状や型を出力する補助メソッドです。
        """
        try:
            batch = next(iter(dataloader))
        except StopIteration:
            print(f"{stage} dataloader にデータがありません。")
            return

        print(f"--- {stage.capitalize()} バッチのサンプル ---")
        if isinstance(batch, dict):
            for key, value in batch.items():
                shape_info = value.shape if hasattr(value, "shape") else type(value)
                print(f"  {key}: {shape_info}")
        elif isinstance(batch, (list, tuple)):
            for idx, item in enumerate(batch):
                shape_info = item.shape if hasattr(item, "shape") else type(item)
                print(f"  Item {idx}: {shape_info}")
        else:
            shape_info = batch.shape if hasattr(batch, "shape") else type(batch)
            print(f"  {shape_info}")
