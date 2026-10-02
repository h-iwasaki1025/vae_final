# 過去の会話・設計・決定事項の全履歴 (Full Conversation History)

このドキュメントは、プロジェクト `vae_final` の立ち上げおよび環境構築における全会話・議論・決定事項をアーカイブしたものです。
他のAIエディタやスマホからのリモートアクセス時にも、このファイルを読み込ませることで全コンテキストを引き継ぐことができます。

---

## 1. UNIX哲学に基づくファイル命名・プロジェクト設計の基本原則

1. **スモール・イズ・ビューティフル (Small is beautiful)**
2. **単一責任の原則 (Do one thing well)**
3. **早めのプロトタイピング (Build a prototype as soon as possible)**
4. **移植性の重視 (Choose portability over efficiency)**
5. **プレーンテキストファースト (Store data in flat text files)**
6. **既存ツールの活用 (Use software leverage)**
7. **シェルスクリプトによる自動化 (Use shell scripts)**
8. **非対話的UI・CLI重視 (Avoid captive user interfaces)**
9. **プログラムをフィルタにする (Make every program a filter)**
10. **沈黙は金なり (Silence is golden)**

---

## 2. ディレクトリ構造の再定義

従来の `pdb_vae_project` （ルート直下に約100ファイル以上が混在）から、UNIX哲学と作業フロー順（`01_`〜`08_`）に沿った新環境 `vae_final` を作成。

* **プロジェクトパス:**
  * ローカル / スマホ共有マウント: `/Volumes/hikaru/Code/vae_final`
  * `cs15` サーバー実機パス: `/Users/hikaru/Code/vae_final`

### 完成した通し番号付きディレクトリ一覧
- `01_config/`: 環境設定・依存ライブラリ管理 (`requirements.txt`, `config.yaml` 等)
- `02_data/`: データ格納 (`raw/`, `interim/`, `processed/`)
- `03_src/`: ソースコード・共通モジュール（再利用可能な関数・モデル定義）
- `04_bin/`: UNIXパイプライン用CLIコマンド・実行ツール
- `05_experiments/`: 学習スクリプト・実験エントリーポイント
- `06_results/`: 評価結果、生成画像、モデル重み (.pt / .ckpt)
- `07_docs/`: Obsidian連動メモ・レポート・引き継ぎ用ドキュメント
- `08_ai_context/`: AIネイティブ開発用ログ・会話履歴・コンテキスト情報
- `README.md`: プロジェクトマップと概要説明

---

## 3. Git および GitHub 連携

* **ローカル Git 初期化:** `.gitignore` を設定し、大容量データ (`02_data/*`) やモデル重み (`*.pt`)、生成画像 (`*.png`, `*.gif`) を自動除外。各空フォルダに `.gitkeep` を配置して構造を保持。
* **GitHub リモートリポジトリ:** `https://github.com/h-iwasaki1025/vae_final` に接続し、`main` ブランチを同期（`git push`）済み。

---

## 4. AIネイティブ開発 ＆ マルチエディタ（ベンダーフリー）対応

* **JSON 形式セッションログ:** `08_ai_context/session_history.json` に過去のアクション・コミット・目的を追記保存。
* **Git による改ざん不可性:** ログを Git コミット・Push することで、過去の決定事項や履歴が消えず、どのAIエディタ（Antigravity, Cursor, Claude Code等）からでも完全に復元・引き継ぎ可能。

---

## 5. cs15 サーバー常時起動 ＆ スマホリモート環境

* **Antigravity CLI (v1.2.14) のインストール:** `/Users/hikarui./.local/bin/agy`
* **常時稼働デーモンの起動:**
  * コマンド: `agy remote-control start --name cs15-server`
  * インスタンス名: `cs15-server`
  * ステータス: `active` （バックグラウンド常時稼働中）
* **スマホアクセス URL:**
  * ダッシュボード: `https://antigravity.google.com` (ログインアカウント: `pineapple20230826@gmail.com`)
  * この会話のダイレクトURL: `https://antigravity.google.com/r/1439e30a-a672-4e2f-91b1-6790f70ff6f3-v2?p=c/671dc40d-1e62-45c4-93c2-00c90daee4c2`

