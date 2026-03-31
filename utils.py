import subprocess
import os


def get_audio_duration(file_path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ],
        stdout=subprocess.PIPE
    )
    return float(result.stdout)


def split_audio(file_path: str, chunk_duration=60):
    """
    使用 ffmpeg 切片
    """
    output_dir = f"/tmp/chunks_{os.path.basename(file_path)}"
    os.makedirs(output_dir, exist_ok=True)

    output_pattern = f"{output_dir}/chunk_%03d.wav"

    subprocess.run([
        "ffmpeg",
        "-i", file_path,
        "-f", "segment",
        "-segment_time", str(chunk_duration),
        "-c", "copy",
        output_pattern
    ])

    chunks = sorted([
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
    ])

    return chunks


def format_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def merge_segments(all_segments):
    """
    修正时间轴
    """
    merged = []
    offset = 0

    for segs in all_segments:
        for seg in segs:
            merged.append({
                "start": seg["start"] + offset,
                "end": seg["end"] + offset,
                "text": seg["text"]
            })
        offset = merged[-1]["end"]

    return merged


def generate_srt(segments):
    srt = ""
    for i, seg in enumerate(segments):
        srt += f"{i+1}\n"
        srt += f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n"
        srt += f"{seg['text'].strip()}\n\n"
    return srt