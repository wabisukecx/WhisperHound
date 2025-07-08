# WhisperHound

A Python tool that finds conversation segments from audio recordings and performs transcription. It leverages Whisper models to support transcription of long-duration audio while filtering out noise and irrelevant sounds to extract only conversations.

## Features

- High-precision transcription using OpenAI Whisper models
- Long-duration audio processing (chunk processing with overlap method)
- Conversation-only extraction (excludes noise, music, and silent segments)
- Filtering of duplicate phrases and meaningless utterances
- Support for multiple output formats (text, JSON, SRT, VTT)
- Both GUI and command-line versions available

## Requirements

- Python 3.8 or higher
- CUDA-compatible GPU recommended (CPU operation also supported)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/wabisukecx/WhisperHound.git
cd whisperhound
```

### 2. Install Required Libraries

```bash
pip install -r requirements.txt
```

Note: For GPU support, please install the appropriate version from the [PyTorch official website](https://pytorch.org/get-started/locally/).

## Usage

WhisperHound provides two interfaces: command-line version and GUI version.

### Command-Line Version Usage

#### Basic Usage

```bash
python main.py -i input.mp3 -o transcript.txt
```

#### Advanced Options

```bash
python main.py -i input.mp3 -o transcript.txt -f json -m large-v3-turbo -l en --chunk-size 300.0
```

### GUI Version (Streamlit App) Usage

```bash
streamlit run whisper_hound_app.py --server.fileWatcherType none
```

A browser will automatically open, providing an intuitive interface where you can:
- Upload audio files
- Select Whisper model and language
- Adjust processing parameters
- Monitor real-time processing status
- Preview and download results

## Command Line Arguments

| Option | Description |
|--------|-------------|
| `-i`, `--input` | Path to input audio file (required) |
| `-o`, `--output` | Path to output file (default: conversation_transcript.txt) |
| `-f`, `--format` | Output format: text, json, srt, vtt (default: text) |
| `-m`, `--model-name` | Whisper model to use (default: large-v3-turbo) |
| `-l`, `--language` | Audio language (default: ja) |
| `--device` | Device to use: cuda, cpu (default: cuda if available) |
| `--prompt` | Prompt to use for transcription |
| `--min-segment-length` | Minimum length of conversation segments in seconds (default: 0.7) |
| `--max-gap` | Maximum gap between segments to merge in seconds (default: 0.5) |
| `--max-repetitions` | Maximum allowed repetitions of the same phrase (default: 2) |
| `--chunk-size` | Size of audio chunks to process at once in seconds (default: 600.0) |
| `--overlap` | Overlap between chunks in seconds (default: 10.0) |
| `--start-time` | Time to start processing in seconds (default: 0.0) |
| `--end-time` | Time to end processing in seconds (default: end of audio) |
| `--append` | Append to existing output file |

## GUI Version Features

The GUI version provides the following additional features:

- Intuitive browser-based interface
- Audio file information display (filename, type, size)
- Automatic audio file conversion option during upload (Whisper optimization)
- Real-time processing status display and progress bar
- Processing result preview display
- One-click result download
- GUI controls for detailed settings

## Project Structure

```
whisperhound/
├── main.py                     # Main script (CLI version)
├── whisper_hound_app.py        # Streamlit app (GUI version)
├── config/
│   └── argument_parser.py      # Command line argument processing
├── utils/
│   ├── audio_utils.py          # Audio file operations
│   ├── time_utils.py           # Timestamp conversion and time-related utilities
│   └── io_utils.py             # File output operations
├── core/
│   ├── model_manager.py        # Whisper model management
│   ├── transcriber.py          # Transcription processing
│   └── conversation_processor.py # Conversation extraction and processing
├── formatters/
│   └── output_formatter.py     # Output format processing
└── convert_audio.py            # Audio conversion utility
```

## Technical Overview

WhisperHound employs a sophisticated multi-stage approach to extract meaningful conversations from audio recordings. The process begins by dividing long audio files into manageable chunks with overlapping segments to ensure no conversation boundaries are lost during processing.

Each audio chunk undergoes transcription using OpenAI's Whisper model, which provides not only the text content but also confidence scores and timing information for each segment. The system then applies intelligent filtering algorithms to distinguish between actual conversation and background noise, music, or other non-conversational audio.

The conversation detection algorithm evaluates multiple factors including segment duration, speech probability scores, and content patterns to identify genuine dialogue. It specifically filters out repeated phrases that often result from audio artifacts or transcription errors, while preserving natural conversational repetitions that carry meaning.

After initial extraction, the system performs segment merging to combine closely spaced utterances that belong to the same conversational flow, creating more natural and readable transcripts. This approach ensures that brief pauses or hesitations don't artificially fragment continuous speech.

The final output undergoes additional processing to remove duplicate content that may arise from the overlapping chunk processing method, ensuring a clean and coherent final transcript.

## Performance Considerations

The tool is designed to handle audio files of varying lengths efficiently. For optimal performance with long recordings, the default chunk size of 10 minutes (600 seconds) with 10-second overlaps provides a good balance between processing speed and transcription accuracy.

GPU acceleration is highly recommended for faster processing, especially when working with larger Whisper models. The system automatically detects CUDA availability and defaults to GPU processing when possible, significantly reducing transcription time for long audio files.

## License

MIT License
