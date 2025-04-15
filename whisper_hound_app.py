#!/usr/bin/env python3
"""
WhisperHound Streamlit App - 音声から会話部分を探し出して文字起こしするツール
"""

import os
import sys
import time
import math
import tempfile
import warnings
import streamlit as st
import numpy as np
from typing import List, Dict, Any, Optional
import torch
import whisper
from pydub import AudioSegment

# Suppress warnings
warnings.filterwarnings("ignore")

# App configuration
st.set_page_config(
    page_title="WhisperHound",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Ported functions from WhisperHound modules ---

# From core/model_manager.py
def load_whisper_model(model_name: str, device: str):
    """
    標準のWhisperモデルをロードします。
    """
    try:
        with st.spinner(f"Whisperモデル '{model_name}' をロード中..."):
            model = whisper.load_model(model_name, device=device)
            return model
    except Exception as e:
        st.error(f"モデルのロード中にエラーが発生: {e}")
        return None

# From utils/audio_utils.py
def load_audio(file_path: str, start: float = 0.0, end: Optional[float] = None) -> np.ndarray:
    """
    音声ファイルを読み込みます。必要に応じてスライスします。
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
                st.warning(f"警告: 開始時間 {start}秒 は音声の長さを超えています")
                return np.array([])
        
        return audio
    except Exception as e:
        st.error(f"音声ファイルの読み込み中にエラーが発生: {e}")
        return np.array([])

def get_audio_duration(file_path: str) -> float:
    """
    音声ファイルの長さを取得します。
    """
    try:
        audio = whisper.load_audio(file_path)
        return len(audio) / 16000  # 16kHzのサンプルレートを仮定
    except Exception as e:
        st.error(f"音声ファイルの長さ取得中にエラーが発生: {e}")
        return 0.0

# From utils/time_utils.py
def format_timestamp_srt(seconds: float) -> str:
    """
    SRTファイル形式のタイムスタンプをフォーマットします。
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_remainder = seconds % 60
    milliseconds = int((seconds_remainder - int(seconds_remainder)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(seconds_remainder):02d},{milliseconds:03d}"

def format_timestamp_vtt(seconds: float) -> str:
    """
    VTTファイル形式のタイムスタンプをフォーマットします。
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_remainder = seconds % 60
    milliseconds = int((seconds_remainder - int(seconds_remainder)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(seconds_remainder):02d}.{milliseconds:03d}"

# From core/transcriber.py
def transcribe_audio(model, audio: np.ndarray, language: Optional[str] = None, 
                     device: str = "cpu", prompt: str = "", offset: float = 0.0) -> Dict[str, Any]:
    """
    Whisperモデルを使用して音声データを文字起こしします。
    """
    try:
        if len(audio) == 0:
            st.warning("警告: 空の音声データ")
            return {"segments": []}
        
        transcription_options = {
            "verbose": False,
            "word_timestamps": True,
            "no_speech_threshold": 0.5,
            "logprob_threshold": -1.0,
            "condition_on_previous_text": False
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
        st.error(f"文字起こし中にエラーが発生: {e}")
        return {"segments": []}

# From core/conversation_processor.py
def is_repeated_phrase(segments: List[Dict[str, Any]], current_idx: int, phrase: str, threshold: int = 3) -> bool:
    """
    直近のセグメントで同じフレーズが繰り返されているかチェックします。
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
            
    return repeat_count >= threshold - 1

def is_conversation(segment: Dict[str, Any], segments: Optional[List[Dict[str, Any]]] = None, 
                   idx: Optional[int] = None, min_segment_length: float = 0.7) -> bool:
    """
    セグメントが会話の一部かどうかを判断します。
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

# From formatters/output_formatter.py
def format_output(segments: List[Dict[str, Any]], format_type: str) -> str:
    """
    会話セグメントを要求された出力形式にフォーマットします。
    """
    import json
    
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

# From convert_audio.py
def convert_to_whisper_wav(input_file, output_file):
    """m4aファイルをWhisper用にwavファイルに変換 (16000Hz, モノラル)"""
    try:
        audio = AudioSegment.from_file(input_file)
        # Whisper推奨設定に変換
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_file, format="wav")
        return True
    except Exception as e:
        st.error(f"エラー: 音声の変換中に問題が発生しました: {e}")
        return False

# Custom functions for the Streamlit app
def process_audio_chunk(model, audio_path, start_time, end_time, config, status_placeholder):
    """音声ファイルの指定された部分を処理"""
    status_placeholder.write(f"音声チャンク {format_timestamp_srt(start_time)} から {format_timestamp_srt(end_time)} までを処理中...")
    
    # 音声データをロード
    audio = load_audio(audio_path, start_time, end_time)
    
    # 文字起こし
    transcription = transcribe_audio(
        model,
        audio,
        language=config["language"],
        device=config["device"],
        prompt=config["prompt"],
        offset=start_time
    )
    
    # 会話セグメントを抽出
    conversation_segments = extract_conversations(transcription, config["min_segment_length"])
    
    if not conversation_segments:
        status_placeholder.write(f"チャンク {format_timestamp_srt(start_time)} - {format_timestamp_srt(end_time)} で会話セグメントが見つかりませんでした")
        return []
    
    status_placeholder.write(f"{len(conversation_segments)} 個の会話セグメントが見つかりました")
    
    # フィルタリング処理
    filtered_segments = filter_repeated_phrases(conversation_segments, config["max_repetitions"])
    filtered_segments = filter_nonsense_phrases(filtered_segments)
    
    # マージ処理
    merged_segments = merge_adjacent_segments(filtered_segments, config["max_gap"])
    
    # 再度フィルタリング
    final_segments = filter_repeated_phrases(merged_segments, config["max_repetitions"])
    
    status_placeholder.write(f"処理後: {len(final_segments)} 個のセグメント")
    
    return final_segments

def process_full_audio(model, audio_path, config, status_placeholder):
    """音声ファイル全体を処理"""
    # 音声の長さを取得
    duration = get_audio_duration(audio_path)
    status_placeholder.write(f"音声ファイルの長さ: {format_timestamp_srt(duration)}")
    
    # 開始・終了時間を設定
    start_time = config.get("start_time", 0.0)
    end_time = config.get("end_time", None) or duration
    
    # チャンクサイズとオーバーラップを設定
    chunk_size = config["chunk_size"]
    overlap = config["overlap"]
    
    # チャンク数を計算
    num_chunks = math.ceil((end_time - start_time) / (chunk_size - overlap))
    status_placeholder.write(f"合計 {num_chunks} チャンクを処理します")
    
    # Progress bar
    progress_bar = st.progress(0)
    
    all_segments = []
    
    # 各チャンクを処理
    for i in range(num_chunks):
        chunk_start = start_time + i * (chunk_size - overlap)
        chunk_end = min(chunk_start + chunk_size, end_time)
        
        # 最後のチャンクの場合は調整
        if chunk_end >= end_time:
            chunk_end = end_time
        
        status_placeholder.write(f"\nチャンク {i+1}/{num_chunks} を処理中...")
        segments = process_audio_chunk(model, audio_path, chunk_start, chunk_end, config, status_placeholder)
        all_segments.extend(segments)
        
        # Update progress bar
        progress_bar.progress((i + 1) / num_chunks)
    
    # 重複するセグメントを除去
    unique_segments = []
    time_tolerance = 0.5
    
    for segment in sorted(all_segments, key=lambda x: x["start"]):
        if not unique_segments:
            unique_segments.append(segment)
            continue
            
        last_segment = unique_segments[-1]
        if (abs(segment["start"] - last_segment["start"]) < time_tolerance and 
            segment["text"].strip().lower() == last_segment["text"].strip().lower()):
            continue
            
        unique_segments.append(segment)
    
    status_placeholder.write(f"\n重複除去後: {len(unique_segments)} 個のセグメント")
    
    # 出力をフォーマット
    formatted_output = format_output(unique_segments, config["output_format"])
    
    return formatted_output, unique_segments

# App title and description
st.title("WhisperHound")
st.subheader("音声から会話部分を探し出して文字起こしするツール")
st.markdown("""
このアプリは音声ファイルをアップロードし、OpenAI Whisperモデルを使用して会話部分のみを
抽出・文字起こしします。長時間の音声にも対応しており、様々な出力形式をサポートしています。
""")

# Sidebar for configuration
with st.sidebar:
    st.header("設定")
    
    # Audio conversion section
    st.subheader("音声変換設定")
    convert_audio = st.checkbox("アップロード後に音声を変換", value=True, 
                                help="音声をWhisper用に最適化 (16000Hz, モノラル) します")
    
    # Model and device selection
    st.subheader("モデル設定")
    available_models = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
    model_name = st.selectbox(
        "Whisperモデル",
        available_models,
        index=5,  # Default to large-v3-turbo
        help="大きいモデルほど精度が高いですが、処理に時間がかかります"
    )
    
    device = st.radio(
        "デバイス",
        ["cuda" if torch.cuda.is_available() else "cpu", "cpu"],
        index=0,
        help="GPUが利用可能な場合はcudaを選択すると処理が高速化されます"
    )
    
    language = st.text_input("言語コード", value="ja", help="例: ja (日本語), en (英語), auto (自動検出)")
    
    # Audio processing settings
    st.subheader("処理設定")
    output_format = st.selectbox(
        "出力形式",
        ["text", "json", "srt", "vtt"],
        index=0
    )
    
    # Advanced settings expander
    with st.expander("詳細設定"):
        prompt = st.text_area("プロンプト", value="これはボードゲームで遊んでいる会話です。", 
                              help="文字起こしの精度向上のためのヒント")
        
        col1, col2 = st.columns(2)
        with col1:
            min_segment_length = st.slider("最小セグメント長（秒）", 0.1, 2.0, 0.7,
                                          help="これより短いセグメントは無視されます")
            max_gap = st.slider("最大間隔（秒）", 0.1, 5.0, 0.5,
                               help="この間隔内のセグメントはマージされます")
        
        with col2:
            max_repetitions = st.slider("最大繰り返し回数", 1, 5, 2,
                                       help="同じフレーズの許容される最大繰り返し回数")
            chunk_size = st.slider("チャンクサイズ（秒）", 60.0, 1200.0, 600.0,
                                  help="一度に処理する音声の長さ")
            
        overlap = st.slider("オーバーラップ（秒）", 1.0, 30.0, 10.0,
                           help="チャンク間のオーバーラップ時間")
        
        # Time range
        st.subheader("処理範囲")
        use_time_range = st.checkbox("特定の時間範囲を処理", value=False)
        if use_time_range:
            time_col1, time_col2 = st.columns(2)
            with time_col1:
                start_time = st.number_input("開始時間（秒）", value=0.0, min_value=0.0)
            with time_col2:
                end_time = st.number_input("終了時間（秒）", value=None)
        else:
            start_time = 0.0
            end_time = None

# Main content area
st.header("音声ファイルのアップロード")
uploaded_file = st.file_uploader("音声ファイルを選択", type=["mp3", "wav", "m4a", "ogg", "flac"])

# Main processing logic
if uploaded_file is not None:
    file_details = {"ファイル名": uploaded_file.name, "ファイルタイプ": uploaded_file.type, "ファイルサイズ": f"{uploaded_file.size / 1e6:.2f} MB"}
    st.write("ファイル情報:", file_details)
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_button = st.button("文字起こしを開始", use_container_width=True)
    
    if start_button:
        status_placeholder = st.empty()
        result_placeholder = st.empty()
        
        with st.spinner("処理中..."):
            try:
                # Create temporary directory for files
                temp_dir = tempfile.TemporaryDirectory()
                temp_input_path = os.path.join(temp_dir.name, uploaded_file.name)
                temp_output_path = os.path.join(temp_dir.name, "processed_audio.wav")
                
                # Save uploaded file
                with open(temp_input_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Convert if needed
                if convert_audio:
                    status_placeholder.write("音声を変換中...")
                    if not convert_to_whisper_wav(temp_input_path, temp_output_path):
                        st.error("音声変換に失敗しました")
                        temp_dir.cleanup()
                        st.stop()
                    audio_path = temp_output_path
                    status_placeholder.success("音声変換が完了しました")
                else:
                    audio_path = temp_input_path
                
                # Load model
                model = load_whisper_model(model_name, device)
                if model is None:
                    st.error("モデルのロードに失敗しました")
                    temp_dir.cleanup()
                    st.stop()
                    
                status_placeholder.success(f"モデルのロードが完了しました - デバイス: {device}")
                
                # Process audio
                config = {
                    "language": None if language == "auto" else language,
                    "device": device,
                    "prompt": prompt,
                    "min_segment_length": min_segment_length,
                    "max_gap": max_gap,
                    "max_repetitions": max_repetitions,
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "output_format": output_format,
                    "start_time": start_time,
                    "end_time": end_time
                }
                
                status_placeholder.write("音声処理を開始します...")
                output, segments = process_full_audio(model, audio_path, config, status_placeholder)
                
                # Display results
                status_placeholder.success("処理が完了しました！")
                
                with result_placeholder.container():
                    st.header("文字起こし結果")
                    
                    # セグメント数表示
                    st.write(f"抽出された会話セグメント: {len(segments)}個")
                    
                    # Display output based on format
                    if output_format == "text":
                        st.text_area("テキスト出力", output, height=300)
                    elif output_format == "json":
                        st.json(segments)
                    elif output_format in ["srt", "vtt"]:
                        st.text_area("字幕出力", output, height=300)
                    
                    # Download button
                    extension = ".txt" if output_format == "text" else f".{output_format}"
                    st.download_button(
                        label=f"{output_format.upper()}ファイルをダウンロード",
                        data=output,
                        file_name=f"transcript{extension}",
                        mime="text/plain"
                    )
                
                # Clean up
                temp_dir.cleanup()
                
            except Exception as e:
                st.error(f"処理中にエラーが発生しました: {str(e)}")
                st.exception(e)
else:
    st.info("音声ファイルをアップロードしてください")

# Footer
st.markdown("---")
st.caption("WhisperHound - 音声録音から会話部分のみを探し出し、文字起こしを行うPythonツール")