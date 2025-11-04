import os
import torch
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed
)
from peft import PeftModel
from tqdm import tqdm
import logging
import argparse
import time
from datetime import datetime

# =================================================================
# 設定パラメータの説明
#
# このスクリプトの全ての動作パラメータは、main() 関数内の
# 'parser.add_argument' セクションでコマンドライン引数として定義される。
#
# [主要な引数とその役割]
#   --lora_path        : 訓練済みLoRAモデルのパス (推論モデルの選択)
#   --dataset_basename : 評価データセットのファイル名 (データソースの選択)
#   --batch_size       : 推論の処理速度とメモリ効率を制御
#   --max_new_tokens   : モデルの回答の最大長を制御
#   --max_length       : プロンプトの最大長を制御 (入力の切り捨てを制御)
#
# これらの引数は、実行時に '--引数名 値' の形式で変更可能です。
# =================================================================


# Hugging Faceトークンを環境変数からロードする（Llama3.1に必須）
# ！！！注意！！！実行前にHF_TOKEN,HUGGINGFACE_TOKEN環境変数を設定するのを忘れない
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

# ロガーを取得する（rootロガーを使用しない）
logger = logging.getLogger(__name__)

# シードの設定
set_seed(42)

# --- 関数: コマンドライン引数解析 ---
def parse_args():
    """コマンドライン引数を解析する"""
    parser = argparse.ArgumentParser(description="Run inference on a pre-trained or base model.")
    
    #  新規追加: ベースモデルID
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct", 
        help="ベースモデルのHugging Face ID。",
    )

    # LoRAアダプターのパス
    parser.add_argument(
        '--lora_path', 
        type=str, 
        default=None, 
        help="LoRAアダプターが保存されているディレクトリのパス。これを指定した場合、ベースモデルにLoRAウェイトが適用されます。Noneの場合はベースモデル単体で推論を実行します。"
    )
    
    # 評価データセットのベース名
    parser.add_argument(
        '--dataset_basename', 
        type=str, 
        default='', 
        help="評価に使用するデータセットのベース名。例: '1000'を指定すると 'data/test_1000.jsonl' を読み込みます。"
    )
    
    # バッチサイズ
    parser.add_argument(
        '--batch_size', 
        type=int,
        default=4, 
        help="推論時のバッチサイズ。一度に処理するプロンプトの数を制御し、メモリ使用量と実行速度に影響します。GPU/MPSのメモリ容量に合わせて調整します。"
    )
    
    # 最大生成トークン数　（プロンプト改良してそれ以外の文言出力しないため、デフォルト値も256でいいはず)
    parser.add_argument(
        '--max_new_tokens', 
        type=int, 
        default=256, 
        help="モデルが生成する新しいトークンの最大数。この値が大きいほど、より長い回答が期待できますが、実行時間も長くなります。"
    )
    
    # 最大入力トークン数
    parser.add_argument(
        '--max_length', 
        type=int, 
        default=1024, 
        help="トークナイズされる入力シーケンス（プロンプト）の最大長。これを超える長さのプロンプトは切り捨てられます。"
    )
    
    # Hugging Faceアクセストークン
    parser.add_argument(
        '--hf_token',
        type=str,
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"), # 環境変数 HF_TOKENまたはHUGGINGFACE_TOKENから読み込む
        help="Hugging Face Hubにアクセスするためのトークン。Llama 3.1のようなゲート付きモデルのロードに必須です。デフォルトでは環境変数 HUGGINGFACE_TOKEN から読み込みます。"
    )
    
    # 出力する推論結果のファイル名
    parser.add_argument(
        '--inference_filename', 
        type=str, 
        default='inference_default.jsonl',
        help="出力する推論結果JSONLのファイル名。デフォルトは 'inference_default.jsonl'"
    )
    
    # ログサフィックス
    parser.add_argument(
        '--log_suffix', 
        type=str, 
        default='',
        help="ログファイル名に付加するカスタムサフィックス。例: '_test_1'を指定すると 'inference_..._timestamp_test_1.log' となる。"
    )

    return parser.parse_args()

# --- 関数: データロード ---
def load_data(file_path):
    """JSONLファイルからプロンプトをロードする"""
    prompts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                # プロンプト部分 (<s>[INST] ... [/INST] の直前まで) を抽出
                # data.get('text', '')がNoneでないことを確認
                text = data.get('text', '')
                if text:
                    prompt_text = text.split('[/INST]')[0] + '[/INST]'
                    prompts.append(prompt_text)
                else:
                    logger.warning(f"行に 'text' キーがない、または空でした。: {line.strip()}")
        return prompts
    except FileNotFoundError:
        logger.error(f"データファイルが見つかりません: {file_path}")
        return []

# --- 関数: 推論実行 ---
# デバイス(device)とデータ型(DTYPE)を引数に追加し、不要な引数を削除
def run_inference(model, tokenizer, dataset, output_file, batch_size, max_new_tokens, max_length, stop_token_ids, device, DTYPE):
    """
    推論を実行し、結果をJSONLファイルに保存する。
    """
    all_inference = []
    
    # datasetオブジェクトからプロンプトのリストを取得し、'prompts'として定義
    prompts = dataset
    
    # データをバッチに分割
    for i in tqdm(range(0, len(prompts), batch_size), desc="Inference"):
        batch_prompts = prompts[i:i + batch_size]
        
        # トークナイズ
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length # 最大入力トークン長
        )
        
        # 入力トークンIDの長さを保持（デコード時にプロンプト部分をスキップするため）
        prompt_lengths = inputs['input_ids'].shape[1]
        
        # 入力をデバイスに転送
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 推論
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens, # 最大生成トークン数
                do_sample=False,
                eos_token_id=stop_token_ids,
                pad_token_id=tokenizer.eos_token_id,
            )

        # デコードと保存
        for j in range(len(batch_prompts)):
            # 生成された回答部分のトークンIDを抽出
            generated_ids = output_ids[j, prompt_lengths:]
            
            # デコード
            prediction_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            logger.info(f" 推論が完了しました。")
            # EOSトークン以降を削除
            if tokenizer.eos_token:
                prediction_text = prediction_text.split(tokenizer.eos_token)[0].strip()
            
        # ---  2つのキーで保存する（INSTがたくさん出ていたため解消  ---

            response = prediction_text
            
            # 1. 最初に出現する [/INST] タグを削除
            #    replace(..., 1)を使用することで、最初に出現した1回だけ削除する
            response = response.replace('[/INST]', '', 1)
            
            # 2. 先頭の改行やスペースをすべて削除し、純粋な回答部分だけを取り出す
            #    これにより、"Joy:..." のように、回答が綺麗に始まる
            pure_response = response.lstrip()
            
            # 3. 末尾の不要な空白や改行も同時に削除 (出力形式の統一)
            final_response =  pure_response.rstrip()
            
            # ------------------------------------------------------------------
            
            # 1. プロンプト: モデルに入力したテキスト ([/INST]まで) を取得
            full_prompt_with_tags = batch_prompts[j].strip()
            
            # 2. [/INST]タグを末尾から削除。純粋な指示文のみにする。
            prompt = full_prompt_with_tags.removesuffix('[/INST]').strip()

            all_inference.append({
                "prompt": prompt,                          # プロンプト ([/INST]まで)
                "response": final_response                # 予測内容 (クリーンアップ済み)
            })
    
        # ---  修正箇所 終了  ---

    # 結果をJSONLファイルに書き出す前に、ディレクトリが存在することを確認する
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in all_inference:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    logger.info(f"推論が完了しました。結果は '{output_file}' に保存されました。")

# --- メイン処理 ---
def main():
    # 実行開始時刻を記録
    start_time = time.time()
    
    # --- 設定パラメータ（コマンドライン引数）---
    args = parse_args() # parse_args() 関数から引数を取得
    # --- コマンドライン引数設定 終了 ---

    # 環境変数からトークンが取得できなかった場合のフォールバック
    hf_token = args.hf_token
    if not hf_token:
        logger.error("エラー：環境変数(HF_TOKEN,HUGGINGFACE_TOKEN)が設定されていません。Llama3.1をロードできません。")
        sys.exit(1) # エラーコードで終了する

    MODEL_ID = args.model_id

    # --- ログ設定　開始 ---
    # ＊ファインチューニング前後で結果を比較するため、モデルタイプによってafter_loraかbaseどちらかにわける
    model_type = "after_lora" if args.lora_path else "base"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR = "logs"
    os.makedirs(LOG_DIR, exist_ok=True)

    # ログファイル名生成ロジック
    LOG_FILE = os.path.join(LOG_DIR, f"{args.log_suffix}_{timestamp}.log")
    
    # ロガーのレベル設定
    logger.setLevel(logging.INFO)
    
    # フォーマッターを定義
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # ファイルハンドラの作成
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.INFO) # INFOで様子見る
    file_handler.setFormatter(formatter)
    
    # ストリームハンドラ（コンソール出力)の作成
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    
    # ハンドラをロガーに追加（シンプル化）
    # ロガーを整理し、以前のif not logger.handlers:を削除した
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    # --- ログ設定 終了 ---

    logger.info("--- 推論開始設定 ---")
    logger.info(f"ログファイル: {LOG_FILE}")

    #  Hugging Faceトークンの設定チェック
    if not args.hf_token:
        logger.error("エラー: 環境変数(HF_TOKEN,HUGGINGFACE_TOKEN)が設定されていません。Llama 3.1をロードできません。")
        sys.exit(1)
    
    # --- 固定設定パラメータ ---
    DATASET_PATH = f'data/llama_test_prompt_{args.dataset_basename}.jsonl'
    
    # inference_filenameを使用
    INFERENCE_DIR = "inference" # ディレクトリ名は固定としておく
    OUTPUT_FILE = os.path.join(INFERENCE_DIR, args.inference_filename)

    # --- 固定設定パラメータ 終了 ---
    
    # --- デバイスとデータ型設定 ---
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
        logger.info(f" MPS (Metal Performance Shaders) が利用可能です。デバイス: {device}")
        DTYPE = torch.float16
    else:
        # CUDAが利用可能な場合はfloat16、そうでなければfloat32
        if torch.cuda.is_available():
            DTYPE = torch.float16
            device = "cuda"
            logger.info(f"CUDA が利用可能です。デバイス: {device}")
        else:
            DTYPE = torch.float32
            logger.info(f"MPS, CUDA は利用できません。デバイス: {device} (CPU)")

    logger.info(f"使用データ型: {DTYPE}")
    logger.info(f"モデルタイプ: {model_type} (LoRAパス: {args.lora_path})")
    logger.info(f"評価データ: {DATASET_PATH}")
    logger.info(f"バッチサイズ: {args.batch_size}, 最大生成トークン数: {args.max_new_tokens}, 最大入力長: {args.max_length}")
    
    # --- モデルとトークナイザーのロード ---
    try:
        logger.info(f"ベースモデル '{MODEL_ID}' を CPU (device_map=None) に {DTYPE} でロード中... (メモリ回避策)")
        
        # 1. ベースモデルをdevice_map=None(CPU)にロード
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map=None, # ここを必ずNoneに変更
            token=hf_token,
            dtype=DTYPE, # ここは dtype ではなく torch_dtype に戻した（HuggingFaceが推奨）
            trust_remote_code=True,
            use_cache=True, # 注意！Gemmaでは必要ないためコメントアウトする
        )
        
        # トークナイザーのロード
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, 
            token=hf_token,
            trust_remote_code=True,
            use_fast=False, # 安定性向上のため
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        logger.info("ベースモデルとトークナイザーのロードが完了しました。")

        # ==========================================================
        # 【STOP SEQUENCEの定義】
        stop_token_ids = [tokenizer.eos_token_id]
        
        #  不要：「投稿日:」を停止トークンに追加。（ノイズ防止）
        #post_date_token_id = tokenizer.encode('投稿日:', add_special_tokens=False)
        #if post_date_token_id:
            # 投稿日: のトークンIDを追加
        #    stop_token_ids.extend(id for id in post_date_token_id if id not in stop_token_ids)
    
        # 最終的な stop_token_ids には </s> と 投稿日:のトークンIDが入る。
    
        # ==========================================================
        # 2. LoRAアダプターのロードと適用
        if args.lora_path:
            if not os.path.exists(args.lora_path):
                 raise FileNotFoundError(f"LoRAアダプターパスが見つかりません：{args.lora_path}")
            logger.info(f"LoRAアダプター ({args.lora_path}) をロード中...")
            
            model = PeftModel.from_pretrained(base_model, args.lora_path)
            # model = model.merge_and_unload() # メモリが非常にタイトな場合はマージなし
            logger.info("LoRAアダプターの適用が完了しました。")
            
        else:
            model = base_model
        
        # 3. モデルを明示的に適切なデバイスに転送（ロード後に転送する）
        # if device == "mps":  
        if device != "cpu":  # 念の為、CUDAとMPSの両方で転送を保証
            model.to(torch.device(device))
            logger.info(f"最終モデルを明示的に {device} デバイスに転送しました。")

        # 推論モードに設定
        model.eval()
        logger.info("モデルのロードと設定が完了しました。")

        # --- データロードと推論実行 ---
        logger.info(f"データセット '{DATASET_PATH}' をロード中...")
        prompts = load_data(DATASET_PATH)
        
        if not prompts:
            logger.error("エラー: ロードされたプロンプトが空です。処理を終了します。")
            return

        logger.info(f"プロンプト数: {len(prompts)}件")
        
        # run_inferenceへの引数の順番を合わせ、device/DTYPEを追加
        run_inference(
            model=model, 
            tokenizer=tokenizer, 
            dataset=prompts, 
            output_file=OUTPUT_FILE, # run_inferenceの定義に合わせる
            batch_size=args.batch_size, 
            max_new_tokens=args.max_new_tokens,
            max_length=args.max_length,
            stop_token_ids=stop_token_ids,
            device=device, # 追加
            DTYPE=DTYPE # 追加
        )

        # 実行時間の計測とログ出力
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"全処理が完了しました。総実行時間：{elapsed_time:.2f}秒({elapsed_time / 60:.2f} 分)")
        
    except Exception as e:
        logger.error(f"致命的なエラーが発生しました。：{e}", exc_info=True)


if __name__ == "__main__":
    main()