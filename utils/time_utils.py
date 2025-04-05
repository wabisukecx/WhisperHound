#!/usr/bin/env python3
"""
WhisperHound - 時間関連のユーティリティ関数を提供するモジュール。
"""


def format_timestamp_srt(seconds: float) -> str:
    """
    SRTファイル形式のタイムスタンプをフォーマットします。
    
    Args:
        seconds: 秒単位の時間
        
    Returns:
        フォーマットされたタイムスタンプ文字列（HH:MM:SS,mmm）
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_remainder = seconds % 60
    milliseconds = int((seconds_remainder - int(seconds_remainder)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(seconds_remainder):02d},{milliseconds:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """
    VTTファイル形式のタイムスタンプをフォーマットします。
    
    Args:
        seconds: 秒単位の時間
        
    Returns:
        フォーマットされたタイムスタンプ文字列（HH:MM:SS.mmm）
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_remainder = seconds % 60
    milliseconds = int((seconds_remainder - int(seconds_remainder)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(seconds_remainder):02d}.{milliseconds:03d}"