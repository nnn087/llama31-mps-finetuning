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
    
    # ハイパーパラメータ
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=700, # 訓練データ確認したら最大でも650くらい、余裕持って700
        help="訓練時の最大入力シーケンス長。推論スクリプトの --max_length に相当。",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1, # 
        help="訓練エポック数。",
    )
    parser.add_argument(
        "--train_limit",
        type=int,
        default=1,
        help="訓練データセットのサンプル制限数。-1を指定すると無制限。",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRAのランク (r)。",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRAのスケーリング係数 (alpha)",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,  
        help="LoRAのドロップアウト率",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1, 
        help="デバイスごとの訓練バッチサイズ。",
    )
    parser.add_argument(
        "--grad_acc_steps",
        type=int,
        default=4,
        help="勾配蓄積ステップ数。",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4, 
        help="学習率。",
    )
    
    # 評価/保存ステップ数
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100, 
        help="評価/保存ステップ数。",
    )

    # シード値
    parser.add_argument( 
        "--seed",
        type=int,
        default=42, 
        help="乱数シード値。実験の再現性確保に使用。",
    )

    # CPUコア数
    parser.add_argument( 
        "--num_workers",
        type=int,
        default=4, 
        help="データセットの前処理（トークナイズ）に使用するCPUコア数。",
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
        default='',
        help="評価に使用するデータセットファイルのパス。",
    )

    # HUGGINGFACEのトークン
    parser.add_argument(
        '--hf_token',
        type=str,
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"),
        help="Hugging Face Hubにアクセスするためのトークン"
    )
    
    # ログディレクトリ
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="ログファイルを保存するディレクトリ。",
    )
    
    # 精度設定 (FP16/BF16)
    parser.add_argument(
        '--fp16',
        action='store_true', # --fp16 が指定されたら True になる
        help="float16 トレーニングを有効にする (MPS推奨)。"
    )
    parser.add_argument(
        '--bf16',
        action='store_true', # --bf16 が指定されたら False のまま
        help="bfloat16 トレーニングを有効にする (CUDA Ampere以降推奨)。"
    )
    
    # デフォルト値を設定
    # MPS (Mac) での実行を想定し、fp16をデフォルトTrueにします
    parser.set_defaults(fp16=True, bf16=False)
    
    return parser.parse_args()

# --- 関数: ロギング設定 ---
def setup_logging(args):
    """ロギング設定を行い、ログファイル名を返す"""
    
    # ログディレクトリの作成
    os.makedirs(args.log_dir, exist_ok=True) 

    # ログファイル名を作成
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(args.log_dir, f"train_{args.target_modules}_{current_time}.log")
    
    # ロガーフォーマッターを定義
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # ファイルハンドラを作成
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # ストリームハンドラ（コンソール出力）を作成 
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter) 
    
    # ロガーのレベル設定
    logger.setLevel(logging.INFO)
    
    # ルートロガーにファイルハンドラとストリームハンドラを追加
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # transformersライブラリのログもキャプチャ
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.INFO)
    transformers_logger.addHandler(file_handler)
    transformers_logger.addHandler(stream_handler)
    
    logger.info(f"ログファイル: {log_filename} に詳細を出力します。")
    
    return log_filename
    
# --- 関数:デバイス設定 ---
def setup_device():
    """実行デバイスを判定し、デバイス名とdevice_mapを返す"""
    if torch.backends.mps.is_available():
        device = "mps" 
        device_map = None # MPSは 'auto' に非対応。CPUロードが必須
        logger.info(f"デバイス: {device} (Metal Performance Shaders が利用可能です)。")
        logger.info("MPSではモデルはCPUにロードされ、float16で訓練されます (device_map=None)。")

    elif torch.cuda.is_available(): # CUDA用
        device = "cuda"
        device_map = "auto" # CUDAは 'auto' を使う
        logger.info(f"デバイス: {device} (CUDA が利用可能です)。")
        logger.info("モデルは CUDAにロードされます (device_map='auto')。")
        
    else:
        device = "cpu" 
        device_map = None
        logger.warning("MPSもCUDAも利用できません。CPUで実行します（非常に時間がかかります）。")
        
    return device, device_map

# --- 関数: モデルとトークナイザのロード、LoRA設定の分離 ---
def load_model_and_tokenizer(args, hf_token, device_map):
    """モデル、トークナイザをロードし、LoRAを適用する"""
    
    effective_batch_size = args.batch_size * args.grad_acc_steps
    target_modules_list = LORA_TARGET_MAP.get(args.target_modules, LORA_TARGET_MAP["qv"])
    
    logger.info(f"--- モデルとトークナイザのロード: {args.model_id} ---")
    logger.info(f"--- LoRAターゲット: {args.target_modules} ({len(target_modules_list)}層) ---")
    logger.info(f"--- ハイパーパラメータ: エポック={args.num_epochs}, 最大系列長={args.max_seq_length}, 実質バッチサイズ={effective_batch_size} ---")
    logger.info(f"--- LoRA設定: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}, LR={args.learning_rate} ---")
    
    # モデルのロード
    logger.info(f"モデルを float16/bfloat16 (fp16={args.fp16}, bf16={args.bf16}) で初期化しロードします...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map=device_map,
            token=hf_token,
            dtype=torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32), 
            trust_remote_code=True,
            use_cache=False, 
        )

    except Exception as e:
        logger.error(f"モデルのロード中に予期せぬエラーが発生しました: {e}")
        logger.error("上記のエラーは、メモリ不足またはHugging Face認証エラーである可能性が高いです。")
        sys.exit(1)

    # メモリ節約のため、勾配チェックポインティングを有効化
    model.gradient_checkpointing_enable()
        
    # トークナイザのロード
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        token= hf_token,
        trust_remote_code=True,
        use_fast=False,
    )

    # Llamaモデルのトークナイザ設定
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    # --- LoRA設定の適用 ---
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules_list,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

# --- 関数: データセットのロードとトークナイズ ---
def prepare_dataset(args, tokenizer):
    """データセットのロード、サンプリング、トークナイズを行う"""
    logger.info("--- データセットのロードとトークナイズ ---")
    try:
        dataset = load_dataset(
            'json',
            data_files={
                'train': args.train_data_file, 
                'eval': args.eval_data_file
            }
        )
        
        # 訓練データが制限数より多い場合のみサンプリング
        if args.train_limit != -1 and "train" in dataset and len(dataset["train"]) > args.train_limit:
            dataset["train"] = dataset["train"].select(range(args.train_limit))
            logger.info(f"訓練データセットを {args.train_limit} 件に絞り込みました。")
        elif "train" in dataset:
            logger.info(f"訓練データセットは {len(dataset['train'])} 件です（train_limit={args.train_limit}）。")

        # 訓練データ例を表示
        if "train" in dataset and len(dataset["train"]) > 0:
            logger.info("訓練データ例:")
            logger.info(dataset["train"][0]["text"])
            
        # データセットを事前にトークナイズする関数
        def tokenize_function(examples):
            return tokenizer(
                examples["text"], 
                truncation=True, 
                max_length=args.max_seq_length, 
                padding="max_length"
            )

        # データセットをトークナイズ（CPU並列処理で効率化）
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"], 
            num_proc=args.num_workers,
        )
        
        # Trainerで使用するために、'labels'カラムを作成
        tokenized_datasets = tokenized_datasets.map(
            lambda examples: {'labels': examples['input_ids']},
            batched=True,
            num_proc=args.num_workers,
        )

    except Exception as e:
        logger.error(f"データセットのロードまたはトークナイズ中にエラーが発生しました: {e}")
        logger.error(f"指定されたデータセットパスを確認してください: Train={args.train_data_file}, Eval={args.eval_data_file}")
        sys.exit(1)
        
    return tokenized_datasets

# --- メイン処理 ---
def main():
    # 0.実行開始時刻を記録
    start_time = time.time()
    
    # 1. コマンドライン引数設定（パラメータ設定）
    args = parse_args()

    # 2. 引数で取得したシード値で乱数を固定
    set_seed(args.seed)

    # 3. 環境変数からトークンが取得できなかった場合のフォールバック
    hf_token = args.hf_token
    if not hf_token:
        logger.error("エラー: 環境変数(HF_TOKEN,HUGGINGFACE_TOKEN)が設定されていません。Llama 3.1をロードできません。")
        sys.exit(1) # エラーコードで終了
    
    # 4. 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)

    # 5. ロギング設定の初期化
    log_filename = setup_logging(args)
    
    # 6. デバイスと環境のチェック
    device, device_map = setup_device()

    # 7. モデルとトークナイザのロード、LoRAの適用
    model, tokenizer = load_model_and_tokenizer(args, args.hf_token, device_map)

    # 8. データセットの準備
    tokenized_datasets = prepare_dataset(args, tokenizer)

    # 9. トレーニング引数の設定
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs, # 引数（エポック数：だいたい１、重いから）
        per_device_train_batch_size=args.batch_size, # 引数（訓練のバッチサイズ）
        per_device_eval_batch_size=args.batch_size, # 引数（評価のバッチサイズ）
        gradient_accumulation_steps=args.grad_acc_steps, # 引数から
        optim="adamw_torch",
        
        # --- ▼ Early Stopping (過学習対策) のための変更　ここから ▼ ---
        # "epoch" ではなく "steps" で評価・保存
        eval_strategy="steps",
        save_strategy="steps",
        
        # 評価・保存の間隔をステップで指定
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,

        # ログもステップに合わせる
        logging_steps=args.eval_steps, 
        
        # 最高のモデルを自動でロードする設定 (最重要)
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss", # eval_loss を監視
        greater_is_better=False,        # loss は低い方が良い

        # ディスク節約（最新のチェックポイントを2つだけ保持）
        save_total_limit=2,

        # ログの出力先を指定
        report_to="all", # "none" から変更
        # --- ▲ Early Stopping (過学習対策) のための変更　ここまで　▲ ---
        
        learning_rate=args.learning_rate, # 引数から
        fp16=args.fp16, 
        bf16=args.bf16, 
        group_by_length=True,
        lr_scheduler_type="cosine",
        disable_tqdm=False, 
        include_inputs_for_metrics=True,
        gradient_checkpointing=True, 
    )

    # 10. 言語モデル用のデータコレーター (パディングとラベル生成を処理)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 11. Trainerの設定と実行
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["eval"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    train_result = trainer.train()

    # 12. 最終モデルとトークナイザの保存
    trainer.model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_checkpoint"))
    
    end_time = time.time()
    logger.info(f"トレーニングが完了しました。モデルは {args.output_dir}/final_checkpoint に保存されました。")
    logger.info(f"合計実行時間: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()
