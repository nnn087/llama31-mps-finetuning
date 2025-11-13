#!/bin/bash
# スクリプトを堅牢にするための設定
# -e: コマンドがエラーになったら即座に終了
# -u: 未定義の変数を使おうとしたらエラー
# -o pipefail: パイプラインの途中でエラーが発生したら、その時点でエラーとする
set -euo pipefail
# --- LoRA トレーニング実行スクリプト ---
#
# 長いPythonコマンドライン引数を省略し、簡単にLoRAトレーニングを実行するためのスクリプトです。
#
# 実行方法:
# 1. ファイルの上部にある 'PARAMETERS' セクションを編集します。
# 2. ターミナルで実行権限を与えます: chmod +x run_training.sh
# 3. 実行します: ./run_training.sh
#
# ===============================================
# PARAMETERS: ここを編集して設定を変更してください
# ===============================================

# ベースモデルID
BASE_MODEL_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"  # Instruct-Tunedモデルを推奨

# LoRAを適用するターゲット層のタイプ ('qv' または 'all')
# 'qv' は q_proj と v_proj のみ
# 'all' は全ての線形層
TARGET_MODULES="all"

# 学習済みモデルの出力ディレクトリ名 (./output/ の下のフォルダ名)
# 例: LoRAのターゲット層の名前を使用する。'qv', 'all' など。
# 基本的にはターゲット層の名前が設定される
OUTPUT_DIR_NAME="llama_${TARGET_MODULES}"

# 修正・新規追加: データセットの件数を定義 
# 処理したいデータセットの件数を設定 (例: 450, 35000)
DATASET_SIZE=5 

# 自動生成: データセットファイル名 (DATASET_SIZEに基づいて自動決定) 
# 'data/' フォルダのパスは 'training.py' 側で調整可能だが、ここでは固定
TRAIN_DATA_FILE="../data/llama_train_${DATASET_SIZE}.jsonl" 
EVAL_DATA_FILE="../data/llama_eval_${DATASET_SIZE}.jsonl"   

# ハイパーパラメータ
MAX_SEQ_LENGTH=600        # 最大系列長 (推論スクリプトに合わせる)
NUM_EPOCHS=1              # エポック数
TRAIN_SIZE_LIMIT=1       # 訓練データ制限 (-1で無制限)
BATCH_SIZE=1              # デバイスごとのバッチサイズ
GRAD_ACC_STEPS=8          # 勾配蓄積ステップ数
LEARNING_RATE=3e-4        # 学習率
LORA_R=16                 # LoRAのランク
LORA_ALPHA=32             # LoRAのアルファ値
LORA_DROPOUT=0.1          # LoRAのドロップアウト

# 追加した設定値
SEED=42                   # 乱数シード値（再現性確保）
NUM_WORKERS=4             # データセット前処理に使用するCPUコア数

# ログ/評価/精度 設定
LOG_DIR="../logs"                  # ログファイルの保存先
FP16_FLAG="--fp16"                # 精度設定フラグ (--fp16 または --bf16、使用しない場合は空欄 "")
# BF16_FLAG=""                    # BF16を使用する場合は "--bf16"

# ===============================================
# 評価ステップ自動計算ブロック 
# ===============================================

# 1. 実質バッチサイズを計算 (整数)
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACC_STEPS))

# 2. 総最適化ステップ数を計算 (整数。切り上げ処理を含む)
# (データ件数 / 実質バッチサイズ) を NUM_EPOCHS 倍する
TOTAL_STEPS=$(( ( (DATASET_SIZE + EFFECTIVE_BATCH_SIZE - 1) / EFFECTIVE_BATCH_SIZE ) * NUM_EPOCHS ))

# 3. 評価ステップを決定 (訓練全体で4回評価を行うように設定)
TARGET_EVAL_COUNT=4
# TOTAL_STEPS / 4 で評価ステップを計算。ただし、計算結果が0の場合は最低1とする。
EVAL_STEPS=$(( TOTAL_STEPS / TARGET_EVAL_COUNT ))
if [ "$EVAL_STEPS" -eq 0 ]; then
    EVAL_STEPS=1
fi

# ===============================================
# EXECUTION: コマンドの実行 (通常は編集不要)
# ===============================================

# Pythonスクリプトに渡す引数を構築
OUTPUT_DIR="../output/${OUTPUT_DIR_NAME}"
echo "--- トレーニング開始 (Arguments Summary) ---"
echo "BASE_MODEL_ID:       $BASE_MODEL_ID"       
echo "TARGET_MODULES:      $TARGET_MODULES"
echo "OUTPUT_DIRECTORY:    $OUTPUT_DIR"
echo "MAX_SEQ_LENGTH:      $MAX_SEQ_LENGTH"
echo "NUM_EPOCHS:          $NUM_EPOCHS"
echo "BATCH_SIZE (eff): $(($BATCH_SIZE * $GRAD_ACC_STEPS))"
echo "LEARNING_RATE:       $LEARNING_RATE"
echo "TRAIN_DATA_FILE:     $TRAIN_DATA_FILE" 
echo "EVAL_DATA_FILE:      $EVAL_DATA_FILE"   
echo "SEED:                $SEED"
echo "NUM_WORKERS:         $NUM_WORKERS"
echo "LOG_DIRECTORY:       $LOG_DIR"       
echo "EVAL_STEPS:          $EVAL_STEPS"    
echo "PRECISION:           $FP16_FLAG"     
echo "-----------------------------------------"

# 環境変数 HF_TOKEN (または HUGGINGFACE_TOKEN) が設定されていることを確認
# Llama 3.1のモデルダウンロードに必要です。
if [ -z "$HF_TOKEN" ] && [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "警告: 環境変数 HF_TOKEN または HUGGINGFACE_TOKEN が設定されていません。"
    echo "     設定しない場合、モデルのダウンロードに失敗する可能性があります。"
fi

# Pythonスクリプトの実行
# scripts/train.pyが引数を自動的に解析します。
python llama_training.py \
    --model_id "$BASE_MODEL_ID" \
    --target_modules "$TARGET_MODULES" \
    --output_dir "$OUTPUT_DIR" \
    --train_data_file "$TRAIN_DATA_FILE" \
    --eval_data_file "$EVAL_DATA_FILE" \
    --max_seq_length $MAX_SEQ_LENGTH \
    --num_epochs $NUM_EPOCHS \
    --train_limit $TRAIN_SIZE_LIMIT \
    --batch_size $BATCH_SIZE \
    --grad_acc_steps $GRAD_ACC_STEPS \
    --learning_rate $LEARNING_RATE \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --seed $SEED \
    --num_workers $NUM_WORKERS \
    --log_dir "$LOG_DIR" \
    --eval_steps $EVAL_STEPS \
    $FP16_FLAG

echo "--- トレーニング処理が完了しました ---"
echo "結果は ${OUTPUT_DIR}/final_checkpoint に保存され、"
echo "ログファイルは ${LOG_DIR}/フォルダに生成されます。"
