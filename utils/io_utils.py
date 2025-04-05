#!/usr/bin/env python3
"""
WhisperHound - ファイル入出力関連のユーティリティ関数。
"""

import sys


def save_output(content: str, output_path: str, append: bool = False) -> None:
    """
    フォーマットされた出力をファイルに保存します。
    
    Args:
        content: 保存するコンテンツ
        output_path: 出力ファイルのパス
        append: 追記モードの場合はTrue、上書きモードの場合はFalse
    """
    try:
        mode = "a" if append else "w"
        with open(output_path, mode, encoding="utf-8") as f:
            f.write(content)
        action = "追記" if append else "保存"
        print(f"会話が {output_path} に{action}されました")
    except Exception as e:
        sys.exit(f"出力の保存中にエラーが発生: {e}")