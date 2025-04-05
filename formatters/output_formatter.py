#!/usr/bin/env python3
"""
WhisperHound - 出力フォーマット処理を行うモジュール。
"""

import json
from typing import List, Dict, Any

from utils.time_utils import format_timestamp_srt, format_timestamp_vtt


def format_output(segments: List[Dict[str, Any]], format_type: str) -> str:
    """
    会話セグメントを要求された出力形式にフォーマットします。
    
    Args:
        segments: 会話セグメントのリスト
        format_type: 出力形式（text, json, srt, vtt）
        
    Returns:
        フォーマットされた出力（文字列）
    """
    if format_type == "json":
        return json.dumps(segments, ensure_ascii=False, indent=2)
        
    elif format_type == "srt":
        srt_output = ""
        for i, segment in enumerate(segments, 1):
            start_time = format_timestamp_srt(segment["start"])
            end_time = format_timestamp_srt(segment["end"])
            srt_output += f"{i}\n{start_time} --> {end_time}\n{segment['text']}\n\n"
        return srt_output
        
    elif format_type == "vtt":
        vtt_output = "WEBVTT\n\n"
        for i, segment in enumerate(segments):
            start_time = format_timestamp_vtt(segment["start"])
            end_time = format_timestamp_vtt(segment["end"])
            vtt_output += f"{start_time} --> {end_time}\n{segment['text']}\n\n"
        return vtt_output
        
    else:  # デフォルトはtext
        return "\n".join(segment["text"] for segment in segments)