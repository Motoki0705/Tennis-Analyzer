import os
import shutil

def collect_py_files(src_dir, dest_dir):
    """
    指定されたフォルダ(src_dir)から.pyファイルを再帰的に探し、
    別のフォルダ(dest_dir)にコピーする関数。
    """
    print("--- .pyファイルの収集を開始します ---")
    
    collected_count = 0
    # os.walkで指定したフォルダ内の全ファイル・サブフォルダを探索
    for dirpath, _, filenames in os.walk(src_dir):
        for filename in filenames:
            # 拡張子が.pyで終わるファイルかチェック
            if filename.endswith('.py'):
                source_path = os.path.join(dirpath, filename)
                destination_path = os.path.join(dest_dir, filename)

                # ファイル名の重複への対処
                if os.path.exists(destination_path):
                    relative_path = os.path.relpath(dirpath, src_dir)
                    prefix = 'root' if relative_path == '.' else relative_path.replace(os.sep, '_')
                    new_filename = f"{prefix}_{filename}"
                    destination_path = os.path.join(dest_dir, new_filename)
                    print(f"⚠️ ファイル名が重複したためリネームします: {new_filename}")

                shutil.copy(source_path, destination_path)
                print(f"📄 コピーしました: {source_path} → {destination_path}")
                collected_count += 1
    
    if collected_count == 0:
        print(f"'{src_dir}' 内に.pyファイルは見つかりませんでした。")
    else:
        print(f"\n✅ .pyファイルの収集が完了しました。合計 {collected_count} 個のファイルを集めました。")
    print("-" * 30)


def save_directory_tree(root_dir, output_file):
    """
    指定されたフォルダ(root_dir)の構造をツリー形式でテキストファイルに出力する関数。
    """
    print("\n--- フォルダ構造のツリー生成を開始します ---")
    tree_lines = []

    def build_tree_recursive(dir_path, prefix=""):
        """ツリー構造を再帰的に構築するヘルパー関数"""
        # フォルダ内のアイテムをソートして取得
        try:
            items = sorted(os.listdir(dir_path))
        except FileNotFoundError:
            return
        
        pointers = ['├── '] * (len(items) - 1) + ['└── ']

        for pointer, item in zip(pointers, items):
            full_path = os.path.join(dir_path, item)
            # フォルダの場合は末尾に / を付ける
            display_name = item + '/' if os.path.isdir(full_path) else item
            tree_lines.append(f"{prefix}{pointer}{display_name}")

            if os.path.isdir(full_path):
                # 次の階層のプレフィックスを決定（│を追加するかスペースにするか）
                extension = '│   ' if pointer == '├── ' else '    '
                build_tree_recursive(full_path, prefix=prefix + extension)

    # ツリーの最上部にルートフォルダ名を追加
    tree_lines.append(f"{os.path.basename(root_dir)}/")
    # 再帰関数を呼び出してツリーを構築
    build_tree_recursive(root_dir)

    # 生成したツリーをファイルに書き込む
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(tree_lines))
        print(f"📁 フォルダ構造を '{output_file}' に保存しました。")
    except IOError as e:
        print(f"エラー: ファイル '{output_file}' の書き込みに失敗しました。詳細: {e}")
    print("-" * 30)


# --- メイン処理 ---
if __name__ == "__main__":
    # --- 設定項目 ---

    # .pyファイルを探す対象のフォルダ（このフォルダとそのサブフォルダが対象になります）
    source_folder = 'src/ball/tests/'  # ← ここに移動元のフォルダパスを記入

    # .pyファイルを集める先のフォルダ（存在しない場合は自動で作成されます）
    destination_folder = 'all_code_collection'  # ← ここに集めたい先のフォルダパスを記入

    # --- 設定はここまで ---
    # source_folderやdestination_folderが設定されているかチェック
    if source_folder == r'C:\your_source_project_folder' or destination_folder == r'C:\python_files_collection':
        print("エラー: スクリプト上部の `source_folder` と `destination_folder` のパスをあなたの環境に合わせて設定してください。")
    else:
        # --- 実行 ---
        
        # 移動先フォルダが存在しない場合は作成
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
            print(f"フォルダ '{destination_folder}' を作成しました。")
        
        # 機能1: .pyファイルを集める
        collect_py_files(source_folder, destination_folder)

        # 機能2: フォルダ構造をツリーで保存する
        tree_output_file = os.path.join(destination_folder, 'directory_tree.txt')
        save_directory_tree(source_folder, tree_output_file)

        print("\n✨ すべての処理が完了しました。")
