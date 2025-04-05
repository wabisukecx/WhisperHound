#!/usr/bin/env python3
"""
WhisperHound - 会話セグメントの抽出と処理を行うモジュール。
"""

from typing import Dict, List, Any, Optional


def is_repeated_phrase(segments: List[Dict[str, Any]], current_idx: int, phrase: str, threshold: int = 3) -> bool:
    """
    直近のセグメントで同じフレーズが繰り返されているかチェックします。
    
    Args:
        segments: セグメントのリスト
        current_idx: 現在のセグメントのインデックス
        phrase: チェックするフレーズ
        threshold: 繰り返しと判断するための閾値
        
    Returns:
        フレーズが繰り返されていると判断される場合はTrue、そうでない場合はFalse
    """
    if current_idx < threshold:
        return False
    
    repeat_count = 0
    for i in range(current_idx-1, current_idx-threshold-1, -1):
        if i < 0:
            break
        prev_text = segments[i].get("text", "").strip().lower()
        if prev_text == phrase:
            repeat_count += 1
            
    return repeat_count >= threshold - 1  # threshold-1回以上繰り返されていたら真


def is_conversation(segment: Dict[str, Any], segments: Optional[List[Dict[str, Any]]] = None, 
                   idx: Optional[int] = None, min_segment_length: float = 0.7) -> bool:
    """
    セグメントが会話の一部かどうかを判断します。
    
    Args:
        segment: Whisperレスポンスからのセグメントデータ
        segments: すべてのセグメントのリスト（繰り返しチェック用）
        idx: 現在のセグメントのインデックス（繰り返しチェック用）
        min_segment_length: 会話と判断するための最小セグメント長（秒）
        
    Returns:
        セグメントが会話の一部である場合はTrue、そうでない場合はFalse
    """
    text = segment.get("text", "").strip().lower()
    
    if not text or len(text) < 2:
        return False
        
    noise_indicators = ["♪", "♫", "[音楽]", "[ノイズ]", "[拍手]", "[笑い]", "[music]", "[noise]", "[applause]", "[laughter]"]
    if any(indicator in text for indicator in noise_indicators):
        return False
    
    # 繰り返しフレーズのチェックを追加
    if segments and idx is not None and idx > 0:
        if is_repeated_phrase(segments, idx, text):
            return False
    
    start = segment.get("start", 0)
    end = segment.get("end", 0)
    if end - start < min_segment_length:
        return False
        
    if "no_speech_prob" in segment and segment["no_speech_prob"] > 0.6:
        return False
            
    return True


def extract_conversations(transcription: Dict[str, Any], min_segment_length: float = 0.7) -> List[Dict[str, Any]]:
    """
    文字起こしから会話セグメントを抽出します。
    
    Args:
        transcription: Whisperからの文字起こしデータ
        min_segment_length: 会話と判断するための最小セグメント長（秒）
        
    Returns:
        テキストとタイムスタンプを含む会話セグメントのリスト
    """
    conversation_segments = []
    segments = transcription.get("segments", [])
    
    for idx, segment in enumerate(segments):
        if is_conversation(segment, segments, idx, min_segment_length):
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            text = segment.get("text", "").strip()
            
            conversation_segments.append({
                "start": start,
                "end": end,
                "text": text
            })
    
    return conversation_segments


def filter_repeated_phrases(segments: List[Dict[str, Any]], max_repetitions: int = 2) -> List[Dict[str, Any]]:
    """
    同じフレーズの連続繰り返しをフィルタリングします。
    
    Args:
        segments: セグメントのリスト
        max_repetitions: 許容する同じフレーズの最大繰り返し回数
        
    Returns:
        フィルタリングされたセグメントのリスト
    """
    if not segments:
        return []
        
    filtered_segments = [segments[0]]
    current_phrase = segments[0]["text"].strip().lower()
    repeat_count = 1
    
    for segment in segments[1:]:
        text = segment["text"].strip().lower()
        if text == current_phrase:
            repeat_count += 1
            if repeat_count <= max_repetitions:
                filtered_segments.append(segment)
        else:
            current_phrase = text
            repeat_count = 1
            filtered_segments.append(segment)
            
    return filtered_segments


def filter_nonsense_phrases(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    意味のないフレーズや繰り返しパターンをフィルタリングします。
    
    Args:
        segments: セグメントのリスト
        
    Returns:
        フィルタリングされたセグメントのリスト
    """
    if not segments:
        return []
    
    # フィルタリングするパターン
    nonsense_patterns = [
        "おやすみなさい",
        "さようなら",
        "あのー",
        "えーと",
        "うーん"
    ]
    
    filtered_segments = []
    
    for segment in segments:
        text = segment["text"].strip()
        # テキストが短すぎる場合や、フィルタリングパターンのみで構成されている場合はスキップ
        if len(text) < 2 or text.lower() in nonsense_patterns:
            continue
        
        filtered_segments.append(segment)
    
    return filtered_segments


def merge_adjacent_segments(segments: List[Dict[str, Any]], max_gap: float = 0.5) -> List[Dict[str, Any]]:
    """
    近接した会話セグメントをマージします。
    
    Args:
        segments: 会話セグメントのリスト
        max_gap: マージするセグメント間の最大時間間隔（秒）
        
    Returns:
        マージされた会話セグメントのリスト
    """
    if not segments:
        return []
        
    sorted_segments = sorted(segments, key=lambda x: x["start"])
    merged_segments = [sorted_segments[0]]
    
    for current in sorted_segments[1:]:
        previous = merged_segments[-1]
        if current["start"] - previous["end"] <= max_gap:
            previous["end"] = current["end"]
            previous["text"] += " " + current["text"]
        else:
            merged_segments.append(current)
    
    return merged_segments