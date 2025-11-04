-----

# Llama 3.1 LoRA ファインチューニング＆推論パイプライン (Apple MPS対応)

## 1\. 概要

本プロジェクトは、`meta-llama/Meta-Llama-3.1-8B-Instruct` モデルに対し、**LoRA (Low-Rank Adaptation)** を用いたファインチューニングおよび推論を実行するためのパイプライン構築事例です。

特に、**Apple Silicon (MPS)** 環境での安定動作と、研究開発における**体系的な実験管理**の実現に重点を置いて設計されています。

> **注記:** 本プロジェクトは研究活動の一環として開発されたものであり、使用した**具体的なタスクやデータセットの詳細**については、**プロジェクトの性質**により、本資料での開示を控えさせていただいております。本ポートフォリオは、タスクの詳細ではなく、実装された**技術的工夫**（環境最適化、実験管理設計、デバッグプロセス）を示すことに焦点を当てています。

-----

## 2\. 本プロジェクトの主な特徴

  * **Apple MPS 環境への最適化**
    Apple Silicon (M1/M2/M3) 環境での学習・推論におけるメモリ不足(**OOM**)や実行時エラーを回避するため、`float16` の明示的な指定、`device_map=None` でのロードと明示的なデバイス転送（`.to(device)`）など、**安定稼働のためのチューニング**を施しています。
  * **柔軟な実験管理手法**
    `run_*.sh` シェルスクリプトを実験の「**設定ファイル**」として機能させ、LoRA適用対象（`TARGET_MODULES`）、データセット規模（`DATASET_SIZE`）、学習率（`LEARNING_RATE`）などの主要パラメータを**変数として一元管理**しています。
  * **再現性と効率化の確保**
    推論スクリプト（`run_inference.sh`）は、実験設定（`METHOD_NAME`など）に基づき、**出力ファイル名やログファイル名を自動生成**します。これにより、結果の上書きを防ぎ、実験結果の比較・管理を容易にしています。
  * **推論時の後処理（ノイズ除去）**
    `llama_inference.py` に、モデルが指示文（`[/INST]`）を繰り返すノイズや不要な空白を自動で除去する**クリーンアップ処理**を実装しています。
  * **ベースライン比較設計**
    `run_inference.sh` で `METHOD_NAME="base"` を指定することで、LoRA適用モデルと**ベースモデル**（事前学習済み）の性能比較実験を**シームレスに切り替え**て実行可能な設計を採用しています。

-----

## 3\. セットアップ

### ステップ 1: リポジトリのクローンと移動

```bash
git clone https://[your-github-username]/[your-repo-name].git
cd [your-repo-name]
```

### ステップ 2: Hugging Face トークンの設定 (必須)

Llama 3.1 モデルへのアクセス権が必要です。スクリプトがトークンの存在をチェックするため、環境変数への設定は必須です。

```bash
# .zshrc や .bashrc に追記を推奨
export HF_TOKEN="hf_YOUR_HUGGING_FACE_TOKEN"
```

### ステップ 3: 依存関係のインストール

`requirements.txt` を使用して必要なライブラリをインストールします。

```bash
pip install -r requirements.txt
```

### ステップ 4: ディレクトリ構成の確認

本スクリプトは以下のディレクトリ構造を前提とします。

```
.
├── data/
│   ├── (学習用データ).jsonl
│   └── (推論用データ).jsonl
│
├── scripts/
│   ├── llama_training.py       <-- Pythonスクリプト
│   └── llama_inference.py      <-- Pythonスクリプト
│
├── llama_run_training.sh       <-- 実行用シェル
├── llama_run_inference.sh      <-- 実行用シェル
│
├── README.md
├── .gitignore
└── requirements.txt
```

> **重要:** `llama_training.py` と `llama_inference.py` は、必ず `scripts/` ディレクトリ内に配置してください。

-----

## 4\. 使用方法

### 学習の実行

1.  `llama_run_training.sh` を開き、**`PARAMETERS` セクション**でハイパーパラメータを設定します。
      * `TARGET_MODULES`: `qv` または `all` を選択
      * `DATASET_SIZE`: `data/` フォルダ内のファイル名に対応する識別子
      * `LORA_R`, `LEARNING_RATE` などを調整
2.  実行権限を付与し、実行します。

<!-- end list -->

```bash
chmod +x llama_run_training.sh
./llama_run_training.sh
```

  * 学習済みモデル（アダプタ）は `output/` に、詳細ログは `logs/` に保存されます。

### 推論の実行

1.  `llama_run_inference.sh` を開き、**`PARAMETERS` セクション**を設定します。
      * `METHOD_NAME`: 学習済みモデル（`all`, `qv`）またはベースモデル（`base`）を選択
      * `DATASET_SIZE`: 推論に使用するファイルの識別子
      * `BATCH_SIZE`: ご自身のMPSメモリ容量に合わせて調整
2.  実行権限を付与し、実行します。

<!-- end list -->

```bash
chmod +x llama_run_inference.sh
./llama_run_inference.sh
```

  * 推論結果（`.jsonl`）は `inference/` に、詳細ログは `logs/` に保存されます。

-----

## 5\. データセットについて

本スクリプトは、Llama 3.1 Instructモデルの学習形式（`<s>[INST] ... [/INST] ... </s>`）に**事前加工された `.jsonl` 形式**のデータセットを読み込むよう設計されています。

研究上の理由により具体的なデータセットの内容は非公開ですが、上記の形式に準拠した任意のテキスト分類・生成タスクのデータセットで動作する**汎用的な設計**となっています。

-----

## 6\. ライセンス

本プロジェクトは **MIT License** のもとで公開されています。

-----