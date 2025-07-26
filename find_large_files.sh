#!/bin/bash

# Gitリポジトリ内の履歴全体から、容量の大きいファイルを特定するスクリプト

# --- 設定 ---
# 検索するファイルの上位件数。スクリプトの第一引数で指定できます。
# 使用例: ./find_large_files.sh 20
COUNT=${1:-10} # 引数がなければデフォルトで10件

# --- 事前チェック ---
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  echo "エラー: このスクリプトはGitリポジトリ内で実行してください。"
  exit 1
fi

# --- メイン処理 ---
echo "🚀 リポジトリの履歴全体から容量の大きいファイルをTOP ${COUNT}件検索します..."
echo "（注意: リポジトリの規模によっては数分以上かかる場合があります）"
echo "------------------------------------------------------------"

# 1. リポジトリ内の全オブジェクトから、サイズの大きいblob（ファイル）を特定
#    - git rev-list: 全コミットの全オブジェクトをリストアップ
#    - git cat-file: オブジェクトの情報を取得
#    - grep: blob（ファイル）のみを抽出
#    - sort: サイズ順（降順）に並べ替え
#    - head: 上位N件を取得
large_objects=$(git rev-list --objects --all |
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize)' |
  grep '^blob' |
  sort -rh -k3 |
  head -n "$COUNT")

# 2. 特定したblobの情報を表示し、関連するファイルパスを検索
echo "$large_objects" | while read -r type sha size; do

  # ファイルサイズを人間が読みやすい形式に変換 (B, KB, MB, GB...)
  size_hr=$(echo "$size" | awk '
    function human(x, s) {
      s="B KB MB GB TB PB"
      while (x >= 1024 && length(s) > 2) {
        x /= 1024
        s = substr(s, 4)
      }
      return sprintf("%.2f %s", x, substr(s, 1, 2))
    }
    { print human($1) }')

  echo "📦 SHA: $sha | Size: $size_hr"

  # 3. 特定したSHAを持つファイルが、どのコミット・パスに存在したかを検索
  #    - git log --all: 全てのブランチ・タグを検索対象にする
  #    - --pretty='%T %h': 各コミットのツリーハッシュとコミットハッシュを取得
  #    - git ls-tree -r: ツリー内を再帰的に検索
  #    - grep: 該当のSHAを持つ行を絞り込む
  #    - awk: 結果を整形して表示
  #    - sort -u: 重複するパス表示をまとめる
  git log --all --pretty=format:'%T %h' | while read tree_hash commit_hash; do
    git ls-tree -r "$tree_hash" 2>/dev/null | grep "$sha" | while read -r _; do
      # grepで見つかった行の最後のフィールド（パス）を取得
      path="${_##*	}"
      echo "  - Path: ${path}  (in commit ${commit_hash})"
    done
  done | sort -u
  
  echo "------------------------------------------------------------"
done

echo "✅ 検索完了"