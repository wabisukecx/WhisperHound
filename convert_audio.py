#!/usr/bin/env python3
"""
m4aファイルをWhisperに最適化したwavファイルに変換するスクリプト
Whisperの推奨設定: 16000Hz, モノラル
"""

import os
import argparse
from pydub import AudioSegment
import glob
from tqdm import tqdm

def convert_to_whisper_wav(input_file, output_file=None):
    """m4aファイルをWhisper用にwavファイルに変換 (16000Hz, モノラル)"""
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + ".wav"
    
    try:
        audio = AudioSegment.from_file(input_file, format="m4a")
        # Whisper推奨設定に変換
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_file, format="wav")
        return output_file
    except Exception as e:
        print(f"エラー: '{input_file}'の変換中に問題が発生しました: {e}")
        return None

def process_directory(input_dir, output_dir=None, recursive=False):
    """ディレクトリ内のすべてのm4aファイルをWhisper用wavに変換"""
    if output_dir is None:
        output_dir = input_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    pattern = os.path.join(input_dir, "**", "*.m4a") if recursive else os.path.join(input_dir, "*.m4a")
    m4a_files = glob.glob(pattern, recursive=recursive)
    
    converted_count = 0
    
    for m4a_file in tqdm(m4a_files, desc="ファイル変換中"):
        rel_path = os.path.relpath(m4a_file, input_dir)
        output_file = os.path.join(output_dir, os.path.splitext(rel_path)[0] + ".wav")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        if convert_to_whisper_wav(m4a_file, output_file):
            converted_count += 1
    
    return converted_count

def main():
    parser = argparse.ArgumentParser(description="m4aファイルをWhisper用wavファイル(16000Hz,モノラル)に変換")
    parser.add_argument("-i", "--input", required=True, help="入力m4aファイルまたはディレクトリ")
    parser.add_argument("-o", "--output", help="出力wavファイルまたはディレクトリ")
    parser.add_argument("--recursive", action="store_true", help="ディレクトリ内のサブディレクトリも処理")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        parser.error(f"入力ファイルまたはディレクトリが存在しません: {args.input}")
    
    if os.path.isdir(args.input):
        count = process_directory(args.input, args.output, args.recursive)
        print(f"{count}個のファイルが変換されました")
    else:
        result = convert_to_whisper_wav(args.input, args.output)
        if result:
            print(f"変換が完了しました: {result}")
        else:
            print("変換に失敗しました")

if __name__ == "__main__":
    main()