#!/usr/bin/env python3
"""
WhisperHound - Whisperモデルの管理を行うモジュール。
"""

import sys
import torch
import whisper


def load_whisper_model(model_name: str, device: str):
    """
    標準のWhisperモデルをロードします。
    
    Args:
        model_name: モデル名
        device: 使用するデバイス（'cuda'または'cpu'）
        
    Returns:
        ロードされたWhisperモデル
    """
    try:
        print(f"Whisperモデル '{model_name}' をロード中...")
        model = whisper.load_model(model_name, device=device)
        return model
    except Exception as e:
        sys.exit(f"モデルのロード中にエラーが発生: {e}")