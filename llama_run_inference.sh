#!/bin/bash
set -e
# --- 推論実行スクリプト ---
#
# 長いPythonコマンドライン引数を省略し、簡単に推論を実行するためのスクリプトです。
#
# 実行方法:
# 1. ファイルの上部にある 'PARAMETERS' セクションを編集します。
# 2. ターミナルで実行権限を与えます: chmod +x run_inference.sh
# 3. 実行します: ./run_inference.sh
#
# ==========================================================
# PARAMETERS: ここを編集して設定を変更する
# ==========================================================

# ベースモデルID
BASE_MODEL_ID="meta-llama/Meta-Llama-3.1-8B-Instruct" 

# 学習/実行方法を示す共通の名前 (例: qv_only, all, base)
# 下記のいずれかを入力する
# qv_only : qv層のみ学習を実施したモデル
# all : 全ての線形層に対して学習を実施モデル
# base : ベースモデル
METHOD_NAME="all"

# LoRAアダプターのパス (自動生成)　＊変えなくていい_出力先変えたら変える
if [ "$METHOD_NAME" = "base" ]; then
    LORA_PATH="" # ベースモデルの場合はパスを空にする
else
    LORA_PATH="./output/llama_${METHOD_NAME}/final_checkpoint"
fi

# 修正・新規追加：データセットの件数を定義
# 評価したいデータセットの件数を設定 （例：45, 35000）
DATASET_SIZE=1000

# 評価に使用するデータセットファイル名 (llama_test_prompt_*.jsonl)
DATASET_FILE="data/llama_test_prompt_${DATASET_SIZE}.jsonl"

# 推論バッチサイズ 
BATCH_SIZE=8

# 最大生成トークン数 (回答の長さ)
MAX_NEW_TOKENS=100

# 最大入力トークン数 (プロンプトの切り捨て)
# 
MAX_LENGTH=600

# 【修正】推論結果ファイルのファイル名全体
# INFERENCE_FILENAME ="inference_qv_only_1000_8b.jsonl" のように自動生成される
INFERENCE_FILENAME="llama_inference_${METHOD_NAME}_${DATASET_SIZE}_${BATCH_SIZE}b.jsonl"

# 【修正】ログファイル名に付加するサフィックス (ログファイル名のベースとなる)
# 結果: "inference_qv_only_1000" となり、これがログ名のプレフィックスになる
LOG_FILENAME_SUFFIX="llama_inference_${METHOD_NAME}_${DATASET_SIZE}_${BATCH_SIZE}b"

# 環境変数 HF_TOKEN (または HUGGINGFACE_TOKEN) が設定されていることを確認
# Llama 3.1のモデルダウンロードに必要です。
if [ -z "$HF_TOKEN" ] && [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "警告: 環境変数 HF_TOKEN または HUGGINGFACE_TOKEN が設定されていません。"
    echo "     設定しない場合、モデルのダウンロードに失敗する可能性があります。"
fi

# ==========================================================
# EXECUTION: コマンドの実行 (通常は編集不要)
# ==========================================================
echo "--- 推論開始 (Arguments Summary) ---"
echo "BASE_MODEL_ID:       $BASE_MODEL_ID"  
echo "METHOD_NAME:         $METHOD_NAME"
echo "LORA_PATH:           $LORA_PATH"
echo "DATASET_NAME (SIZE): $DATASET_SIZE"
echo "BATCH_SIZE:          $BATCH_SIZE"
echo "MAX_NEW_TOKENS:      $MAX_NEW_TOKENS"
echo "MAX_LENGTH:          $MAX_LENGTH"
echo "INFERENCE_FILENAME:  $INFERENCE_FILENAME"
echo "LOG_FILENAME_SUFFIX: $LOG_FILENAME_SUFFIX" 
echo "------------------------------------"

python scripts/llama_inference.py \
    --model_id "$BASE_MODEL_ID" \
    --lora_path "$LORA_PATH" \
    --dataset_basename "$DATASET_SIZE" \
    --batch_size $BATCH_SIZE \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_length "$MAX_LENGTH" \
    --inference_filename "$INFERENCE_FILENAME" \
    --log_suffix "$LOG_FILENAME_SUFFIX" 

echo "--- 推論処理が完了しました ---"