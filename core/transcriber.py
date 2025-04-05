#!/usr/bin/env python3
"""
WhisperHound - 音声の文字起こし処理を行うモジュール。
"""

import sys
import torch
import numpy as np
from typing import Dict, Any, Optional


def transcribe_audio(model, audio: np.ndarray, language: Optional[str] = None, 
                     device: str = "cpu", prompt: str = "", offset: float = 0.0) -> Dict[str, Any]:
    """
    Whisperモデルを使用して音声データを文字起こしします。
    
    Args:
        model: Whisperモデル
        audio: 音声データ
        language: 音声の言語コード（Noneの場合は自動検出）
        device: 処理に使用するデバイス（'cuda'または'cpu'）
        prompt: 文字起こしに使用するプロンプト
        offset: タイムスタンプのオフセット（秒）
        
    Returns:
        セグメント付きの文字起こしデータを含む辞書
    """
    try:
        print(f"音声データを文字起こし中...")
        
        if len(audio) == 0:
            print("警告: 空の音声データ")
            return {"segments": []}
        
        transcription_options = {
            "verbose": True,
            "word_timestamps": True,
            "no_speech_threshold": 0.5,  # 無音検出の閾値を上げる
            "logprob_threshold": -1.0,   # 低確率の予測を除外
            "condition_on_previous_text": False  # 前のテキストに条件付けない（チャンク処理のため）
        }
        
        if language:
            transcription_options["language"] = language
            
        if prompt:
            transcription_options["initial_prompt"] = prompt
        
        with torch.no_grad():
            result = model.transcribe(audio, **transcription_options)
        
        # オフセットを追加
        if offset > 0:
            for segment in result.get("segments", []):
                if "start" in segment:
                    segment["start"] += offset
                if "end" in segment:
                    segment["end"] += offset
                    
        return result
    
    except Exception as e:
        sys.exit(f"文字起こし中にエラーが発生: {e}")