#!/usr/bin/env python3
"""
WhisperHound - 音声ファイルの操作に関連するユーティリティ関数。
"""

import sys
import numpy as np
from typing import Optional
import whisper


def load_audio(file_path: str, start: float = 0.0, end: Optional[float] = None) -> np.ndarray:
    """
    音声ファイルを読み込みます。必要に応じてスライスします。
    
    Args:
        file_path: 音声ファイルのパス
        start: 開始時間（秒）
        end: 終了時間（秒）
        
    Returns:
        読み込まれた音声データ
    """
    try:
        audio = whisper.load_audio(file_path)
        sample_rate = 16000  # Whisperのサンプルレート
        
        if end is not None:
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            if end_sample > len(audio):
                end_sample = len(audio)
            audio = audio[start_sample:end_sample]
        elif start > 0:
            start_sample = int(start * sample_rate)
            if start_sample < len(audio):
                audio = audio[start_sample:]
            else:
                print(f"警告: 開始時間 {start}秒 は音声の長さを超えています")
                return np.array([])
        
        return audio
    except Exception as e:
        sys.exit(f"音声ファイルの読み込み中にエラーが発生: {e}")


def get_audio_duration(file_path: str) -> float:
    """
    音声ファイルの長さを取得します。
    
    Args:
        file_path: 音声ファイルのパス
        
    Returns:
        音声の長さ（秒）
    """
    try:
        audio = whisper.load_audio(file_path)
        return len(audio) / 16000  # 16kHzのサンプルレートを仮定
    except Exception as e:
        sys.exit(f"音声ファイルの長さ取得中にエラーが発生: {e}")