import argparse
import os

from faster_whisper import WhisperModel

STRIP_PUNCTUATION = set('"#$%&()*+,-./:;<=>@[\\]^_`{|}~')
SENTENCE_ENDINGS = {'.', '!', '?'}
ABBREVIATIONS = {'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'sr.', 'jr.', 'vs.', 'etc.', 'inc.', 'ltd.', 'st.', 'ave.', 'blvd.'}
BREAK_BEFORE = {'and', 'but', 'or', 'because', 'although', 'if', 'when', 'while', 'after', 'before'}

_model = None


def get_model(model_size="large-v3-turbo"):
    global _model
    if _model is None:
        print(f"Loading model {model_size}...")
        _model = WhisperModel(model_size, device="auto", compute_type="auto")
    return _model


def is_sentence_end(word):
    lower = word.lower()
    if lower in ABBREVIATIONS:
        return False
    return any(word.endswith(p) for p in SENTENCE_ENDINGS)


def format_timestamp(secs):
    if secs is None:
        return "00:00:00,000"
    hours = int(secs // 3600)
    minutes = int((secs % 3600) // 60)
    secs_remainder = secs % 60
    milliseconds = int((secs_remainder % 1) * 1000)
    seconds = int(secs_remainder)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def transform_word(word, keep_punctuation=False, to_lower=False):
    if not keep_punctuation:
        word = ''.join(c for c in word if c not in STRIP_PUNCTUATION)
    if to_lower:
        word = word.lower()
    return word


def find_best_split_point(words, max_chars):
    """Find the best index to split a list of word dicts, respecting max_chars."""
    text = ''
    last_fits = 0
    break_points = []  # (index, char_count) of potential break points

    for i, w in enumerate(words):
        word = w['display']
        test = text + (' ' if text else '') + word

        if len(test) <= max_chars:
            last_fits = i + 1
            text = test

            # Track potential break points (but not if current word is also a break word)
            word_lower = word.lower().strip('.,!?')
            if word_lower not in BREAK_BEFORE:
                if i + 1 < len(words):
                    next_lower = words[i + 1]['display'].lower().strip('.,!?')
                    if next_lower in BREAK_BEFORE:
                        break_points.append((i + 1, len(text)))
                if word.endswith(','):
                    break_points.append((i + 1, len(text)))
        else:
            # Exceeded max_chars - find best break point
            if break_points:
                # Prefer break points that leave at least 30% on each side
                min_chars = max_chars * 0.3
                good_breaks = [(idx, chars) for idx, chars in break_points
                               if chars >= min_chars and idx <= last_fits]
                if good_breaks:
                    # Take the latest good break (closest to limit without going over)
                    return good_breaks[-1][0]
                # Otherwise take any valid break
                valid = [(idx, chars) for idx, chars in break_points if idx <= last_fits]
                if valid:
                    return valid[-1][0]
            return last_fits if last_fits > 0 else 1

    return len(words)


def split_long_segment(segment, max_chars):
    """Split a segment that exceeds max_chars into smaller pieces."""
    words = segment['words']

    # If it fits, return as-is
    if len(segment['text']) <= max_chars:
        return [segment]

    # If only one word, can't split further
    if len(words) <= 1:
        return [segment]

    results = []
    remaining = words[:]

    while remaining:
        # Check if remaining fits
        remaining_text = ' '.join(w['display'] for w in remaining)
        if len(remaining_text) <= max_chars:
            results.append({
                'text': remaining_text,
                'start_time': remaining[0]['start_time'],
                'end_time': remaining[-1]['end_time'],
                'words': remaining
            })
            break

        # Find best split point
        split_idx = find_best_split_point(remaining, max_chars)

        if split_idx == 0:
            split_idx = 1  # Force at least one word

        left = remaining[:split_idx]
        remaining = remaining[split_idx:]

        results.append({
            'text': ' '.join(w['display'] for w in left),
            'start_time': left[0]['start_time'],
            'end_time': left[-1]['end_time'],
            'words': left
        })

    return results


def create_srt_segments(words, max_chars_per_line, pause_threshold=0.4, keep_punctuation=False, to_lower=False):
    # Phase 1: Build segments using natural breaks only (pause + sentence)
    segments = []
    current_words = []

    for i, word_data in enumerate(words):
        raw_word = word_data['word'].strip()
        display_word = transform_word(raw_word, keep_punctuation, to_lower)
        if not display_word:
            continue

        word_entry = {
            'raw': raw_word,
            'display': display_word,
            'start_time': word_data['start_time'],
            'end_time': word_data['end_time']
        }

        should_break = False

        if current_words:
            # 1. Pause detection
            gap = word_data['start_time'] - current_words[-1]['end_time']
            if gap >= pause_threshold:
                should_break = True

            # 2. Sentence boundary
            if not should_break and is_sentence_end(current_words[-1]['raw']):
                should_break = True

        if should_break and current_words:
            segments.append({
                'text': ' '.join(w['display'] for w in current_words),
                'start_time': current_words[0]['start_time'],
                'end_time': current_words[-1]['end_time'],
                'words': current_words
            })
            current_words = []

        current_words.append(word_entry)

    if current_words:
        segments.append({
            'text': ' '.join(w['display'] for w in current_words),
            'start_time': current_words[0]['start_time'],
            'end_time': current_words[-1]['end_time'],
            'words': current_words
        })

    # Phase 2: Split any segments that exceed max_chars
    split_segments = []
    for seg in segments:
        split_segments.extend(split_long_segment(seg, max_chars_per_line))

    # Phase 3: Merge orphans (1-2 word segments) into previous segment
    # But don't merge if: starts with capital (new sentence) or would exceed 1.5x max
    final_segments = []
    for seg in split_segments:
        word_count = len(seg['words'])
        first_raw = seg['words'][0]['raw'] if seg['words'] else ''
        starts_sentence = first_raw and first_raw[0].isupper()

        can_merge = (
            word_count <= 2
            and final_segments
            and not starts_sentence
            and len(final_segments[-1]['text']) + len(seg['text']) + 1 <= max_chars_per_line * 1.5
        )

        if can_merge:
            prev = final_segments[-1]
            prev['text'] += ' ' + seg['text']
            prev['end_time'] = seg['end_time']
            prev['words'] = prev['words'] + seg['words']
        else:
            final_segments.append(seg)

    # Extend each subtitle's end time to the next subtitle's start time (no gaps)
    for i in range(len(final_segments) - 1):
        final_segments[i]['end_time'] = final_segments[i + 1]['start_time']

    # Generate SRT output
    lines = []
    for i, seg in enumerate(final_segments, 1):
        start = format_timestamp(seg['start_time'])
        end = format_timestamp(seg['end_time'])
        lines.append(f"{i}\n{start} --> {end}\n{seg['text']}\n")

    return '\n'.join(lines)


def process_audio_files(max_chars_per_line, keep_punctuation=False, to_lower=False, language="en", pause_threshold=0.4):
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
            segments, _ = model.transcribe(
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
                words_with_timestamps.append({
                    'word': word_info.word.strip(),
                    'start_time': word_info.start,
                    'end_time': word_info.end
                })

        if not words_with_timestamps:
            print(f"No valid timestamps found for {filename}")
            continue

        base_name = os.path.splitext(filename)[0]
        srt_path = os.path.join('./output', f"{base_name}_c{max_chars_per_line}_p{pause_threshold}.srt")

        try:
            srt_content = create_srt_segments(
                words_with_timestamps,
                max_chars_per_line,
                pause_threshold,
                keep_punctuation,
                to_lower
            )
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            print(f"Saved: {srt_path}")
        except Exception as e:
            print(f"Error generating SRT for {filename}: {e}")
            continue


def parse_args():
    parser = argparse.ArgumentParser(description="Transcribe audio files to SRT subtitles")
    parser.add_argument("chars", type=int, nargs="?", default=60, help="Max characters per subtitle line (default: 60)")
    parser.add_argument("-P", "--proper", action="store_true", help="Enable punctuation and proper capitalization")
    parser.add_argument("-l", "--lang", type=str, default=None, help="Language code (auto-detect if not specified)")
    parser.add_argument("-p", "--pause", type=float, default=0.5, help="Pause threshold in seconds (default: 0.5)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    keep_punctuation = args.proper
    to_lower = not args.proper
    process_audio_files(args.chars, keep_punctuation, to_lower, args.lang, args.pause)
