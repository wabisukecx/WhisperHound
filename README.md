# WhisperHound

音声録音から会話部分のみを探し出し、文字起こしを行うPythonツールです。Whisperモデルを活用して長時間音声の文字起こしをサポートし、雑音や無関係な音を除外して、会話のみを抽出します。

## 特徴

- OpenAI Whisperモデルを使用した高精度な文字起こし
- 長時間音声の処理（チャンク処理とオーバーラップ方式）
- 会話部分のみの抽出（雑音、音楽、無言部分を除外）
- 重複フレーズや意味のないフレーズのフィルタリング
- 複数の出力フォーマットサポート（テキスト、JSON、SRT、VTT）

## 必要条件

- Python 3.8以上
- CUDA対応GPUの使用を推奨（CPU動作も可能）

## インストール

### 1. リポジトリのクローン

```bash
git clone https://github.com/yourusername/whisperhound.git
cd whisperhound
```

### 2. 必要なライブラリのインストール

```bash
pip install -r requirements.txt
```

注意: GPUサポートについては、[PyTorch公式サイト](https://pytorch.org/get-started/locally/)から適切なバージョンをインストールしてください。

## 使用方法

### 基本的な使用法

```bash
python main.py -i input.mp3 -o transcript.txt
```

### 詳細なオプション

```bash
python main.py -i input.mp3 -o transcript.txt -f json -m large-v3-turbo -l ja --chunk-size 300.0
```

### コマンドライン引数

| オプション | 説明 |
|------------|------|
| `-i`, `--input` | 入力音声ファイルのパス（必須） |
| `-o`, `--output` | 出力ファイルのパス（デフォルト: conversation_transcript.txt） |
| `-f`, `--format` | 出力形式: text, json, srt, vtt（デフォルト: text） |
| `-m`, `--model-name` | 使用するWhisperモデル（デフォルト: large-v3-turbo） |
| `-l`, `--language` | 音声の言語（デフォルト: ja） |
| `--device` | 使用するデバイス: cuda, cpu（デフォルト: 利用可能ならcuda） |
| `--prompt` | 文字起こしに使用するプロンプト |
| `--min-segment-length` | 会話セグメントの最小長さ（秒、デフォルト: 0.7） |
| `--max-gap` | マージするセグメント間の最大間隔（秒、デフォルト: 0.5） |
| `--max-repetitions` | 許容する同じフレーズの最大繰り返し回数（デフォルト: 2） |
| `--chunk-size` | 一度に処理する音声チャンクのサイズ（秒、デフォルト: 600.0） |
| `--overlap` | チャンク間のオーバーラップ（秒、デフォルト: 10.0） |
| `--start-time` | 処理を開始する時間（秒、デフォルト: 0.0） |
| `--end-time` | 処理を終了する時間（秒、デフォルト: 音声の終わりまで） |
| `--append` | 既存の出力ファイルに追記する |

## プロジェクト構造

```
whisperhound/
├── main.py                     # メインスクリプト
├── config/
│   └── argument_parser.py      # コマンドライン引数処理
├── utils/
│   ├── audio_utils.py          # 音声ファイル操作関連
│   ├── time_utils.py           # タイムスタンプ変換など時間関連
│   └── io_utils.py             # ファイル出力関連
├── core/
│   ├── model_manager.py        # Whisperモデル管理
│   ├── transcriber.py          # 文字起こし処理
│   └── conversation_processor.py # 会話抽出・処理
└── formatters/
    └── output_formatter.py     # 出力フォーマット処理
```

## ライセンス

MITライセンス
