# media-to-srt

Transcribe audio files to SRT subtitles using Whisper.

## Features

- Word-level timestamps for accurate subtitle sync
- Smart segmentation: breaks at natural pauses and sentence boundaries
- Splits long lines at conjunctions (and, but, if, when, etc.) not mid-phrase
- Orphan prevention: merges 1-2 word fragments into previous line
- Continuous subtitles: no gaps between captions
- Auto language detection
- Supports CUDA, MPS (Apple Silicon), and CPU

## Installation

```bash
uv sync
```

## Usage

Place audio files in `./input/`, then run:

```bash
uv run python transcribe-me.py
```

Default output: lowercase, no punctuation (clean caption style).

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `chars` | 60 | Max characters per subtitle line (positional) |
| `-P, --proper` | off | Enable punctuation and proper capitalization |
| `-l, --lang` | auto | Language code (auto-detected if not specified) |
| `-p, --pause` | 0.5 | Pause threshold (seconds) for segment breaks |

### Examples

```bash
# Default (60 chars, lowercase, no punctuation)
uv run python transcribe-me.py

# 40 character lines
uv run python transcribe-me.py 40

# Proper capitalization and punctuation
uv run python transcribe-me.py 50 -P

# Force Spanish language
uv run python transcribe-me.py 60 -l es

# Short lines with tight pause breaks
uv run python transcribe-me.py 35 -p 0.3
```

## Output

SRT files are saved to `./output/` with format: `{input_name}_c{chars}_p{pause}.srt`

## Model

Uses `large-v3-turbo` via faster-whisper.
