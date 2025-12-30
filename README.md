# media-to-srt

Transcribe audio files to SRT subtitles using Whisper.

## Features

- Word-level timestamps for accurate subtitle sync
- Smart segmentation: breaks at natural pauses and sentence boundaries
- Splits long lines at conjunctions (and, but, if, when, etc.) not mid-phrase
- Orphan prevention: merges 1-2 word fragments into previous line
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

That's it. Sensible defaults are applied.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--chars` | 60 | Max characters per subtitle line |
| `--pause` | 0.5 | Pause threshold (seconds) for segment breaks |
| `--no-punctuation` | off | Strip punctuation from output |
| `--lowercase` | off | Convert all text to lowercase |
| `--lang` | en | Language code |

### Examples

```bash
# Use defaults (60 chars, punctuation kept, 0.5s pause)
uv run python transcribe-me.py

# Shorter lines for mobile
uv run python transcribe-me.py --chars 42

# More aggressive breaks at pauses
uv run python transcribe-me.py --pause 0.3

# Strip punctuation and lowercase (for lyric-style captions)
uv run python transcribe-me.py --no-punctuation --lowercase
```

## Output

SRT files are saved to `./output/` with format: `{input_name}_c{chars}_p{pause}.srt`

## Model

Uses `large-v3-turbo` via faster-whisper.
