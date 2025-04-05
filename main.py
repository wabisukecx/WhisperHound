#!/usr/bin/env python3
"""
WhisperHound - 音声から会話部分を探し出して文字起こしするツール

標準のWhisperモデル（large-v3-turbo）を使用して
音声録音から会話部分のみを抽出するスクリプト。
複数回の実行をサポートし、長時間音声の文字起こしを完了させる。
"""

import sys
import time
import math
from typing import List, Dict, Any
import warnings
warnings.filterwarnings("ignore")

from config.argument_parser import parse_arguments
from utils.audio_utils import load_audio, get_audio_duration
from core.model_manager import load_whisper_model
from core.transcriber import transcribe_audio
from core.conversation_processor import (
    extract_conversations,
    filter_repeated_phrases,
    filter_nonsense_phrases,
    merge_adjacent_segments
)
from utils.time_utils import format_timestamp_srt
from formatters.output_formatter import format_output
from utils.io_utils import save_output


def process_audio_chunk(model, audio_path: str, start_time: float, end_time: float, 
                        args) -> List[Dict[str, Any]]:
    """
    音声ファイルの指定された部分を処理します。
    
    Args:
        model: Whisperモデル
        audio_path: 音声ファイルのパス
        start_time: 開始時間（秒）
        end_time: 終了時間（秒）
        args: コマンドライン引数
        
    Returns:
        処理されたセグメントのリスト
    """
    print(f"音声チャンク {format_timestamp_srt(start_time)} から {format_timestamp_srt(end_time)} までを処理中...")
    
    # 音声データをロード
    audio = load_audio(audio_path, start_time, end_time)
    
    # 文字起こし
    transcription = transcribe_audio(
        model,
        audio,
        language=args.language,
        device=args.device,
        prompt=args.prompt,
        offset=start_time  # タイムスタンプにオフセットを追加
    )
    
    # 会話セグメントを抽出
    conversation_segments = extract_conversations(transcription, args.min_segment_length)
    
    if not conversation_segments:
        print(f"チャンク {format_timestamp_srt(start_time)} - {format_timestamp_srt(end_time)} で会話セグメントが見つかりませんでした")
        return []
    
    print(f"{len(conversation_segments)} 個の会話セグメントが見つかりました")
    
    # フィルタリング処理
    filtered_segments = filter_repeated_phrases(conversation_segments, args.max_repetitions)
    filtered_segments = filter_nonsense_phrases(filtered_segments)
    
    # マージ処理
    merged_segments = merge_adjacent_segments(filtered_segments, args.max_gap)
    
    # 再度フィルタリング
    final_segments = filter_repeated_phrases(merged_segments, args.max_repetitions)
    
    print(f"処理後: {len(final_segments)} 個のセグメント")
    
    return final_segments


def process_full_audio(model, audio_path: str, args) -> None:
    """
    音声ファイル全体を処理します。
    
    Args:
        model: Whisperモデル
        audio_path: 音声ファイルのパス
        args: コマンドライン引数
    """
    # 音声の長さを取得
    duration = get_audio_duration(audio_path)
    print(f"音声ファイルの長さ: {format_timestamp_srt(duration)}")
    
    # 開始・終了時間を設定
    start_time = args.start_time
    end_time = args.end_time if args.end_time is not None else duration
    
    # チャンクサイズとオーバーラップを設定
    chunk_size = args.chunk_size
    overlap = args.overlap
    
    # チャンク数を計算
    num_chunks = math.ceil((end_time - start_time) / (chunk_size - overlap))
    print(f"合計 {num_chunks} チャンクを処理します")
    
    all_segments = []
    
    # 各チャンクを処理
    for i in range(num_chunks):
        chunk_start = start_time + i * (chunk_size - overlap)
        chunk_end = min(chunk_start + chunk_size, end_time)
        
        # 最後のチャンクの場合は調整
        if chunk_end >= end_time:
            chunk_end = end_time
        
        print(f"\nチャンク {i+1}/{num_chunks} を処理中...")
        segments = process_audio_chunk(model, audio_path, chunk_start, chunk_end, args)
        all_segments.extend(segments)
        
        # チャンク間隔を保つために少し待機
        if i < num_chunks - 1:
            print("次のチャンクに進む前に2秒待機中...")
            time.sleep(2)
    
    # 重複するセグメントを除去
    unique_segments = []
    time_tolerance = 0.5  # 0.5秒以内のタイムスタンプは同じとみなす
    
    for segment in sorted(all_segments, key=lambda x: x["start"]):
        if not unique_segments:
            unique_segments.append(segment)
            continue
            
        last_segment = unique_segments[-1]
        # 開始時間が近く、テキストが類似している場合はスキップ
        if (abs(segment["start"] - last_segment["start"]) < time_tolerance and 
            segment["text"].strip().lower() == last_segment["text"].strip().lower()):
            continue
            
        unique_segments.append(segment)
    
    print(f"\n重複除去後: {len(unique_segments)} 個のセグメント")
    
    # 出力をフォーマット
    formatted_output = format_output(unique_segments, args.format)
    
    # 出力を保存
    save_output(formatted_output, args.output, args.append)


def main() -> None:
    """WhisperHoundの会話抽出プロセスを実行するメイン関数。"""
    args = parse_arguments()
    
    model = load_whisper_model(args.model_name, args.device)
    
    print(f"デバイス: {args.device}")
    print(f"音声ファイル: {args.input}")
    
    process_full_audio(model, args.input, args)


if __name__ == "__main__":
    main()