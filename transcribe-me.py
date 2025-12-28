import argparse
import os
import re

from faster_whisper import WhisperModel

STRIP_PUNCTUATION = set('!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~')
SENTENCE_ENDINGS = re.compile(r'[.!?]$')
ABBREVIATIONS = {'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'sr.', 'jr.', 'vs.', 'etc.', 'inc.', 'ltd.', 'st.', 'ave.', 'blvd.'}

_model = None


def get_model(model_size="large-v3-turbo"):
    global _model
    if _model is None:
        print(f"Loading model {model_size}...")
        _model = WhisperModel(model_size, device="auto", compute_type="auto")
    return _model


def is_sentence_end(word):
    if word.lower() in ABBREVIATIONS:
        return False
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

    lines = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp(seg['start_time'])
        end = format_timestamp(seg['end_time'])
        lines.append(f"{i}\n{start} --> {end}\n{seg['text']}\n")

    return '\n'.join(lines)


def process_audio_files(max_chars_per_line, keep_punctuation=False, to_lower=False, language="en"):
    os.makedirs('./input', exist_ok=True)
    os.makedirs('./output', exist_ok=True)

    files = [f for f in os.listdir('./input') if os.path.isfile(os.path.join('./input', f))]
    if not files:
        print("No files found in ./input")
        return

    model = get_model()

    for filename in files:
        input_path = os.path.join('./input', filename)
        print(f"\nProcessing {filename}...")

        try:
            segments, info = model.transcribe(
                input_path,
                language=language,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

        words_with_timestamps = []
        for segment in segments:
            if segment.words is None:
                continue
            for word_info in segment.words:
                raw_text = word_info.word.strip()
                text = raw_text
                if not keep_punctuation:
                    text = ''.join(c for c in text if c not in STRIP_PUNCTUATION)
                if to_lower:
                    text = text.lower()
                words_with_timestamps.append({
                    'word': text,
                    'raw': raw_text,
                    'start_time': word_info.start,
                    'end_time': word_info.end
                })

        if not words_with_timestamps:
            print(f"No valid timestamps found for {filename}")
            continue

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
    parser.add_argument("--lang", type=str, default="en", help="Language code (default: en)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_audio_files(args.max_chars, args.punctuation, args.lowercase, args.lang)
