#!/usr/bin/env python3
"""
WhisperHound - コマンドライン引数の解析を行うモジュール。
"""

import os
import sys
import argparse
import torch
from typing import Optional


def parse_arguments() -> argparse.Namespace:
    """コマンドライン引数を解析します。"""
    parser = argparse.ArgumentParser(
        description="WhisperHound - Whisperモデルを使用して音声ファイルから会話部分を探し出して文字起こしします"
    )
    parser.add_argument(
        "-i", "--input", 
        required=True, 
        help="入力音声ファイルのパス"
    )
    parser.add_argument(
        "-o", "--output", 
        default="conversation_transcript.txt", 
        help="出力トランスクリプトファイルのパス（デフォルト: conversation_transcript.txt）"
    )
    parser.add_argument(
        "-f", "--format", 
        choices=["text", "json", "srt", "vtt"], 
        default="text", 
        help="出力形式（デフォルト: text）"
    )
    parser.add_argument(
        "-m", "--model-name",
        default="large-v3-turbo",
        help="使用するWhisperモデル名（デフォルト: large-v3-turbo）"
    )
    parser.add_argument(
        "-l", "--language",
        default="ja",  # 言語をデフォルトで日本語に設定
        help="音声の言語（例: 'ja', 'en'）。デフォルトは日本語"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="使用するデバイス（例: 'cuda', 'cpu'）。デフォルトはCUDAが利用可能ならcuda、そうでなければcpu"
    )
    parser.add_argument(
        "--prompt", 
        default="これはボードゲームで遊んでいる会話です。", 
        help="文字起こしに使用するプロンプト（オプション）"
    )
    parser.add_argument(
        "--min-segment-length", 
        type=float, 
        default=0.7,  # 最小セグメント長を0.7秒に増やして短い誤認識を減らす
        help="会話と判断するための最小セグメント長（秒、デフォルト: 0.7）"
    )
    parser.add_argument(
        "--max-gap", 
        type=float, 
        default=0.5,  # 2.0から0.5に戻す
        help="マージするセグメント間の最大時間間隔（秒、デフォルト: 0.5）"
    )
    parser.add_argument(
        "--max-repetitions",
        type=int,
        default=2,  # 繰り返し許容回数を減らす
        help="許容する同じフレーズの最大繰り返し回数（デフォルト: 2）"
    )
    parser.add_argument(
        "--chunk-size",
        type=float,
        default=600.0,  # 10分=600秒のチャンクサイズ
        help="一度に処理する音声チャンクのサイズ（秒、デフォルト: 600.0）"
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=10.0,  # 10秒のオーバーラップ
        help="チャンク間のオーバーラップ（秒、デフォルト: 10.0）"
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=0.0,
        help="処理を開始する時間（秒、デフォルト: 0.0）"
    )
    parser.add_argument(
        "--end-time",
        type=float,
        default=None,
        help="処理を終了する時間（秒、デフォルト: 音声の終わりまで）"
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="既存の出力ファイルに追記する（デフォルト: 上書き）"
    )
    
    args = parser.parse_args()
    
    # 入力ファイルの存在チェック
    if not os.path.isfile(args.input):
        sys.exit(f"エラー: 入力ファイル '{args.input}' が存在しません")
    
    return args