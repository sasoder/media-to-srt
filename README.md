# media-to-srt

Transcribe audio files to SRT subtitles using OpenAI's Whisper model.

## Features

- Word-level timestamps for accurate subtitle sync
- Sentence boundary detection (new sentences start new captions)
- Named entity capitalization via spaCy NER
- Configurable character limit per subtitle line
- Optional punctuation preservation
- Supports CUDA, MPS (Apple Silicon), and CPU

## Installation

```bash
uv sync
```

## Usage

Place audio files in `./input/`, then run:

```bash
uv run python transcribe-me.py <max_chars> [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `max_chars` | Maximum characters per subtitle line |
| `--punctuation` | Keep punctuation in output |
| `--lowercase` | Convert all text to lowercase (skips NER) |

### Examples

```bash
# Basic usage - 40 chars per line, no punctuation, NER capitalization
uv run python transcribe-me.py 40

# Keep punctuation
uv run python transcribe-me.py 40 --punctuation

# Lowercase output
uv run python transcribe-me.py 40 --lowercase

# Both flags
uv run python transcribe-me.py 60 --punctuation --lowercase
```

## Output

SRT files are saved to `./output/` with the naming format: `{input_name}_{max_chars}.srt`

## Model

Uses `openai/whisper-large-v3-turbo` for transcription.
