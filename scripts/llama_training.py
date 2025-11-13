import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from accelerate.utils import set_seed
import logging
import argparse 
import sys 
from datetime import datetime 
import time

# ロガーの取得 (rootロガーを使用しない)
logger = logging.getLogger(__name__)

# --- 定数設定 ---
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LOGS_DIR = "./logs"

# Hugging Faceトークンを環境変数からロードする（Llama3.1に必須）
# ！！！注意！！！実行前にHF_TOKEN,HUGGINGFACE_TOKEN環境変数を設定するのを忘れない
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

# 削除: ハードコードされていた定数群を削除または引数のデフォルト値へ移動

# bitsandbytesを使用しない。float16（MPSで最も安定する）を有効化
FP16 = True
BF16 = False 

# シード値の固定
set_seed(42)

# LoRAのターゲットモジュールを定義（辞書型で設定を保持）
# all：全部の線形層
# qv：qv層のみ
LORA_TARGET_MAP = {
    "all": [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj",
    ],
    "qv": [
        "q_proj", "v_proj",
    ]
}

# --- 関数: コマンドライン引数解析 ---
# =====================================================================
    # コマンドライン引数リスト (デフォルト値)
    #
    # [モデル・データ関連]
    #   --model_id (default: "meta-llama/Meta-Llama-3.1-8B-Instruct"): ベースモデルのID
    #   --output_dir (default: "./output/qv_only_test"): 学習済みモデルの保存先
    #   --train_data_file (default: ''): 訓練データセットのパス
    #   --eval_data_file (default: ''): 評価データセットのパス
    #   --hf_token (default: (環境変数)): Hugging Faceトークン
    #
    # [LoRA設定]
    #   --target_modules (default: "qv"): LoRAを適用する層 ('qv' or 'all')
    #   --lora_r (default: 8): LoRAのランク (r)
    #   --lora_alpha (default: 16): LoRAのスケーリング係数 (alpha)
    #   --lora_dropout (default: 0.05): LoRAのドロップアウト率
    #
    # [訓練ハイパーパラメータ]
    #   --max_seq_length (default: 700): 最大入力シーケンス長
    #   --num_epochs (default: 1): 訓練エポック数
    #   --batch_size (default: 1): バッチサイズ
    #   --grad_acc_steps (default: 4): 勾配蓄積ステップ数
    #   --learning_rate (default: 2e-4): 学習率
    #   --train_limit (default: 1): 訓練データ制限数 (-1で無制限)
    # =====================================================================

def parse_args():
    """コマンドライン引数を解析する"""
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning script for Llama 3.1 on MPS.")
    
    # ベースモデルID
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct", # デフォルト値を設定
        help="ベースモデルのHugging Face ID。",
    )

    # 学習結果の出力先
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/qv_only_test",
        help="学習済みモデルを保存するディレクトリ",
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        default="qv",
        choices=['qv', 'all'],
        help="LoRAを適用するターゲット層 ('qv'または'all')",
    )
    
    # 新規追加：ハードコードされていたハイパーパラメータ
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=700, # 元のハードコード値　訓練データ確認したら最大でも650くらい、余裕持って700
        help="訓練時の最大入力シーケンス長。推論スクリプトの --max_length に相当。",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1, # 元のハードコード値
        help="訓練エポック数 (NUM_TRAIN_EPOCHS)。",
    )
    parser.add_argument(
        "--train_limit",
        type=int,
        default=1, # 元のハードコード値
        help="訓練データセットのサンプル制限数 (TRAIN_SIZE_LIMIT)。-1を指定すると無制限。",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8, # 元のハードコード値
        help="LoRAのランク (r)。",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16, # 元のハードコード値
        help="LoRAのスケーリング係数 (alpha)",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05, # 元のハードコード値
        help="LoRAのドロップアウト率",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1, # 元のハードコード値
        help="デバイスごとの訓練バッチサイズ (BATCH_SIZE)。",
    )
    parser.add_argument(
        "--grad_acc_steps",
        type=int,
        default=4, # 元のハードコード値
        help="勾配蓄積ステップ数 (GRADIENT_ACCUMULATION_STEPS)。",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4, # 元のハードコード値
        help="学習率 (LEARNING_RATE)。",
    )
    
    # 訓練データセットのファイルパス
    parser.add_argument(
        "--train_data_file",
        type=str,
        default='', # デフォルト値
        help="訓練に使用するデータセットファイルのパス。",
    )
    
    # 評価データセットのファイルパス
    parser.add_argument(
        "--eval_data_file",
        type=str,
        default='', # デフォルト値
        help="評価に使用するデータセットファイルのパス。",
    )

    # HUGGINGFACEのトークン
    parser.add_argument(
        '--hf_token',
        type=str,
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"),
        help="Hugging Face Hubにアクセスするためのトークン"
    )

    return parser.parse_args()

# --- メイン処理 ---
def main():
    # 実行開始時刻を記録
    start_time = time.time()
    
    # --- コマンドライン引数設定（パラメータ設定）開始　---
    args = parse_args() # parse_args() 関数から引数を取得
    # --- コマンドライン引数設定（パラメータ設定）終了　---

    # 環境変数からトークンが取得できなかった場合のフォールバック
    hf_token = args.hf_token
    if not hf_token:
        logger.error("エラー: 環境変数(HF_TOKEN,HUGGINGFACE_TOKEN)が設定されていません。Llama 3.1をロードできません。")
        sys.exit(1) # エラーコードで終了

    # コマンドライン引数からベースモデル、訓練結果、訓練対象（すべての線形層かqv層だけか）を設定
    MODEL_ID = args.model_id
    OUTPUT_DIR = args.output_dir
    TARGET_MODULES_KEY = args.target_modules

    # コマンドライン引数からハイパーパラメータを取得
    MAX_SEQ_LENGTH = args.max_seq_length
    NUM_TRAIN_EPOCHS = args.num_epochs
    TRAIN_SIZE_LIMIT = args.train_limit
    LORA_R = args.lora_r
    LORA_ALPHA = args.lora_alpha
    LORA_DROPOUT = args.lora_dropout
    BATCH_SIZE = args.batch_size
    GRADIENT_ACCUMULATION_STEPS = args.grad_acc_steps
    LEARNING_RATE = args.learning_rate
    
    # 出力ディレクトリの作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- ログ設定　開始 ---
    # ログディレクトリの作成
    os.makedirs(LOGS_DIR, exist_ok=True) 

    # ログファイルの設定と追加 (File Handler)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(LOGS_DIR, f"train_{TARGET_MODULES_KEY}_{current_time}.log")
    
    # 1. ロガーフォーマッターを定義 (ファイルとストリームの両方で使用)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 2. ファイルハンドラを作成
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # 3. ストリームハンドラ（コンソール出力）を作成 
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter) 
    
    # ロガーのレベル設定
    logger.setLevel(logging.INFO)

    # ルートロガーにファイルハンドラとストリームハンドラを追加
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # 新規追加: transformersライブラリのログもファイルにキャプチャする
    # これにより、Trainerが出力する {'train_loss': ...} などがログファイルに残る
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.INFO)
    transformers_logger.addHandler(file_handler)

    # --- ログ設定　終了 ---

    logger.info("--- トレーニング開始設定 ---")
    logger.info(f"ログファイル: {log_filename} に詳細を出力します。")
    
    # Hugging Faceトークンの設定チェック
    if not hf_token:
        logger.error("エラー: 環境変数(HF_TOKEN,HUGGINGFACE_TOKEN)が設定されていません。Llama 3.1をロードできません。")
        sys.exit(1)

    # --- 1. デバイスと環境のチェック ---
    logger.info("--- 1. デバイスと環境のチェック ---")
    if torch.backends.mps.is_available():
        DEVICE = "mps" # MPSのとき
        device_map = None # MPSは 'auto' に非対応。CPUロードが必須
        logger.info(f"デバイス: {DEVICE} (Metal Performance Shaders が利用可能です)。")
        logger.info("モデルは安定性のため float16 で、CPUにロードされます (device_map=None)。")

    elif torch.cuda.is_available(): # CUDA用
        DEVICE = "cuda"
        device_map = "auto" # ★ CUDAは 'auto' を使う
        logger.info(f"デバイス: {DEVICE} (CUDA が利用可能です)。")
        logger.info("モデルは float16 で、CUDAにロードされます (device_map='auto')。")
        
    else:
        DEVICE = "cpu" # CPUのとき　＊時間がかかったので中止したほうがいいかもしれない
        device_map = None
        logger.info("MPSは利用できません。CPUで実行します（非常に時間がかかります）。")
        
    # 実質的なバッチサイズを計算
    effective_batch_size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    target_modules_list = LORA_TARGET_MAP.get(TARGET_MODULES_KEY, LORA_TARGET_MAP["qv"])

    logger.info(f"\n--- 2. モデルとトークナイザのロード: {MODEL_ID} ---")
    logger.info(f"--- LoRAターゲット: {TARGET_MODULES_KEY} ({len(target_modules_list)}層) ---")
    
    # ログ出力のハイパーパラメータを引数から取得した値に変更
    logger.info(f"--- ハイパーパラメータ: エポック={NUM_TRAIN_EPOCHS}, 最大系列長={MAX_SEQ_LENGTH}, 実質バッチサイズ={effective_batch_size} ---")
    logger.info(f"--- LoRA設定: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}, LR={LEARNING_RATE} ---")
    
    # モデルのロード
    logger.info(f"モデルをfloat16で初期化し、device_map=None (CPUロード) でロードします...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map=device_map,
            token=hf_token,
            dtype=torch.float16, 
            trust_remote_code=True,
            use_cache=False, # llamaの場合は使用
        )

    except Exception as e:
        logger.error(f"モデルのロード中に予期せぬエラーが発生しました: {e}")
        logger.error("上記のエラーは、メモリ不足またはHugging Face認証エラーである可能性が高いです。")
        sys.exit(1)

    # メモリ節約のため、勾配チェックポインティングを有効化
    model.gradient_checkpointing_enable()
        
    # トークナイザのロード
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token= hf_token,
        trust_remote_code=True,
        use_fast=False,
    )

    # --- 3. LoRA設定の適用 ---
    # Llamaモデルのトークナイザ設定
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    logger.info("--- 3. LoRA設定の適用 ---")
    # 変更：LoraConfigに引数から取得した値を設定
    peft_config = LoraConfig(
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        r=LORA_R,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules_list,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # --- 4. データセットのロードとトークナイズ ---
    logger.info("--- 4. データセットのロードとトークナイズ ---")
    try:

        dataset = load_dataset(
            'json',
            data_files={
                'train': args.train_data_file, 
                'eval': args.eval_data_file
            }
        )
        
        # 訓練データが制限数より多い場合のみサンプリング（TRAIN_SIZE_LIMITを使用）
        if TRAIN_SIZE_LIMIT != -1 and len(dataset["train"]) > TRAIN_SIZE_LIMIT:
            dataset["train"] = dataset["train"].select(range(TRAIN_SIZE_LIMIT))
            logger.info(f"訓練データセットを {TRAIN_SIZE_LIMIT} 件に絞り込みました。")
        else:
            logger.info(f"訓練データセットは {len(dataset['train'])} 件です（TRAIN_SIZE_LIMIT={TRAIN_SIZE_LIMIT}）。")

        # 訓練データ例を表示
        logger.info("訓練データ例:")
        logger.info(dataset["train"][0]["text"])
        
        # データセットを事前にトークナイズする関数
        def tokenize_function(examples):
            return tokenizer(
                examples["text"], 
                truncation=True, 
                # 変更: MAX_SEQ_LENGTHを引数から取得した値に変更
                max_length=MAX_SEQ_LENGTH, 
                padding="max_length"
            )

        # データセットをトークナイズ（CPU並列処理で効率化）
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"], 
            num_proc=4,
        )
        
        # Trainerで使用するために、'labels'カラムを作成
        tokenized_datasets = tokenized_datasets.map(
            lambda examples: {'labels': examples['input_ids']},
            batched=True,
            num_proc=4,
        )

    except Exception as e:
        logger.error(f"データセットのロードまたはトークナイズ中にエラーが発生しました: {e}")
        # 修正: エラーメッセージも汎用的に変更
        logger.error(f"指定されたデータセットパスを確認してください: Train={args.train_data_file}, Eval={args.eval_data_file}")
        sys.exit(1)


    # --- 5. トレーニング引数の設定 ---
    logger.info("\n--- 5. トレーニング引数の設定 ---") 
    # 変更: TrainingArgumentsに引数から取得した値を設定
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS, # 引数（エポック数：だいたい１、重いから）
        per_device_train_batch_size=BATCH_SIZE, # 引数（訓練のバッチサイズ）
        per_device_eval_batch_size=BATCH_SIZE, # 引数（評価のバッチサイズ）
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, # 引数から
        optim="adamw_torch",
        save_strategy="epoch",
        logging_steps=25,
        learning_rate=LEARNING_RATE, # 引数から
        fp16=FP16, 
        bf16=BF16, 
        group_by_length=True,
        lr_scheduler_type="cosine",
        disable_tqdm=False, 
        report_to="none", 
        include_inputs_for_metrics=True,
        gradient_checkpointing=True, 
    )

    # 言語モデル用のデータコレーター (パディングとラベル生成を処理)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- 6. Trainerの設定と実行 ---
    logger.info("--- 6. Trainerの設定と実行 ---")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["eval"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # --- 7. トレーニング開始 ---
    logger.info("--- 7. トレーニング開始 ---")
    train_result = trainer.train()

    # 最終モデルとトークナイザの保存
    trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "final_checkpoint"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_checkpoint"))
    logger.info(f"トレーニングが完了しました。モデルは {OUTPUT_DIR}/final_checkpoint に保存されました。")

if __name__ == "__main__":
    main()
