import argparse
import os
import re

# Punctuation to strip (excludes apostrophe for contractions)
STRIP_PUNCTUATION = set('!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~')

import soundfile as sf
import spacy
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

MODEL_ID = "openai/whisper-large-v3-turbo"
SENTENCE_ENDINGS = re.compile(r'[.!?]$')

# Lazy-loaded globals
_pipe = None
_nlp = None


def get_device():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def get_pipe():
    global _pipe
    if _pipe is None:
        device, dtype = get_device()
        print(f"Loading model on {device}...")

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID,
            dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="eager"
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(MODEL_ID)

        _pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            dtype=dtype,
            device=device,
            return_timestamps=True
        )
    return _pipe


def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def is_sentence_end(word):
    return bool(SENTENCE_ENDINGS.search(word))


def format_timestamp(secs):
    if secs is None:
        return "00:00:00,000"
    hours = int(secs // 3600)
    minutes = int((secs % 3600) // 60)
    secs_remainder = secs % 60
    milliseconds = int((secs_remainder % 1) * 1000)
    seconds = int(secs_remainder)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def capitalize_entities_batch(words_data):
    """Capitalize named entities in batch for efficiency."""
    nlp = get_nlp()
    texts = [w['word'] for w in words_data]
    full_text = ' '.join(texts)

    doc = nlp(full_text)
    entity_spans = {(ent.start_char, ent.end_char) for ent in doc.ents}

    result = list(full_text)
    for start, end in entity_spans:
        if start < len(result) and result[start].isalpha():
            result[start] = result[start].upper()

    capitalized_full = ''.join(result)

    # Split back into words preserving original boundaries
    pos = 0
    for w in words_data:
        word_len = len(w['word'])
        w['word'] = capitalized_full[pos:pos + word_len]
        pos += word_len + 1  # +1 for space


def create_srt_segments(words, max_chars_per_line):
    segments = []
    current = {
        'text': '',
        'start_time': None,
        'end_time': None,
        'words': [],
        'raw_words': []
    }

    for word_data in words:
        word = word_data['word'].strip()
        raw = word_data.get('raw', word).strip()
        if not word:
            continue

        if not current['words']:
            current['start_time'] = word_data['start_time']
            current['text'] = word
            current['words'].append(word)
            current['raw_words'].append(raw)
            current['end_time'] = word_data['end_time']
            continue

        test_text = current['text'] + ' ' + word
        prev_raw = current['raw_words'][-1] if current['raw_words'] else ''

        if is_sentence_end(prev_raw) or len(test_text) > max_chars_per_line:
            segments.append(current)
            current = {
                'text': word,
                'start_time': word_data['start_time'],
                'end_time': word_data['end_time'],
                'words': [word],
                'raw_words': [raw]
            }
        else:
            current['text'] = test_text
            current['words'].append(word)
            current['raw_words'].append(raw)
            current['end_time'] = word_data['end_time']

    if current['words']:
        segments.append(current)

    # Build SRT content efficiently
    lines = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp(seg['start_time'])
        end = format_timestamp(seg['end_time'])
        lines.append(f"{i}\n{start} --> {end}\n{seg['text']}\n")

    return '\n'.join(lines)


def process_audio_files(max_chars_per_line, keep_punctuation=False, to_lower=False):
    os.makedirs('./input', exist_ok=True)
    os.makedirs('./output', exist_ok=True)

    files = [f for f in os.listdir('./input') if os.path.isfile(os.path.join('./input', f))]
    if not files:
        print("No files found in ./input")
        return

    pipe = get_pipe()

    for filename in files:
        input_path = os.path.join('./input', filename)
        print(f"\nProcessing {filename}...")

        try:
            audio, samplerate = sf.read(input_path)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        try:
            result = pipe(
                {"raw": audio, "sampling_rate": samplerate},
                chunk_length_s=30,
                stride_length_s=5,
                return_timestamps="word",
                generate_kwargs={
                    "task": "transcribe",
                    "return_legacy_cache": False
                }
            )
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

        if not isinstance(result, dict):
            print(f"Unexpected result type for {filename}")
            continue

        # Extract words with timestamps
        words_with_timestamps = []
        for chunk in result.get("chunks", []):
            timestamp = chunk.get('timestamp', [None, None])
            if timestamp[0] is not None and timestamp[1] is not None:
                raw_text = chunk.get('text', '').strip()
                text = raw_text
                if not keep_punctuation:
                    text = ''.join(c for c in text if c not in STRIP_PUNCTUATION)
                if to_lower:
                    text = text.lower()
                words_with_timestamps.append({
                    'word': text,
                    'raw': raw_text,
                    'start_time': timestamp[0],
                    'end_time': timestamp[1]
                })

        if not words_with_timestamps:
            print(f"No valid timestamps found for {filename}")
            continue

        # Batch capitalize entities (only if not lowercasing)
        if not to_lower:
            capitalize_entities_batch(words_with_timestamps)

        # Generate SRT
        base_name = os.path.splitext(filename)[0]
        srt_path = os.path.join('./output', f"{base_name}_{max_chars_per_line}.srt")

        try:
            srt_content = create_srt_segments(words_with_timestamps, max_chars_per_line)
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            print(f"Saved: {srt_path}")
        except Exception as e:
            print(f"Error generating SRT for {filename}: {e}")
            continue


def parse_args():
    parser = argparse.ArgumentParser(description="Transcribe audio files to SRT subtitles")
    parser.add_argument("max_chars", type=int, help="Maximum characters per subtitle line")
    parser.add_argument("--punctuation", action="store_true", help="Keep punctuation in output")
    parser.add_argument("--lowercase", action="store_true", help="Convert all text to lowercase")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_audio_files(args.max_chars, args.punctuation, args.lowercase)
