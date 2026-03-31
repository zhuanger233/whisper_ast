import os
import json
import re
import whisper
from openai import OpenAI
from worker import celery
from scheduler import acquire_slot, release_slot
from utils import generate_srt, split_audio, merge_segments, get_audio_duration
from redis_client import set_job
from s3_client import download_file, upload_file

models = {}


def get_model(name):
    if name not in models:
        models[name] = whisper.load_model(name, device="cuda")
    return models[name]


def _extract_json_array(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n", "", text)
        if text.endswith("```"):
            text = text[:-3].strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON array found in OpenAI response")
    return json.loads(text[start:end + 1])


def translate_segments_openai(segments, target_language: str):
    client = OpenAI()
    translated = []
    batch_size = 50

    for i in range(0, len(segments), batch_size):
        batch = segments[i:i + batch_size]
        items = [{"text": s["text"]} for s in batch]
        input_json = json.dumps(items, ensure_ascii=False)

        response = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a subtitle translation engine. "
                        "Translate each item to the target language. "
                        "Return ONLY a JSON array of strings in the same order, "
                        "with no extra text."
                    )
                },
                {
                    "role": "user",
                    "content": f"Target language: {target_language}\nInput JSON:\n{input_json}"
                }
            ]
        )

        output_text = response.output_text
        translated_texts = _extract_json_array(output_text)
        if len(translated_texts) != len(batch):
            raise ValueError("Translated items count mismatch")

        for seg, new_text in zip(batch, translated_texts):
            translated.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": new_text
            })

    return translated


@celery.task(bind=True)
def transcribe_task(self, job_id, s3_key, model_name, translate=False, target_language=None):

    if not acquire_slot(model_name):
        raise Exception("GPU busy")

    local_path = f"/tmp/{job_id}.input"

    try:
        # =========================
        # 从 S3 下载
        # =========================
        set_job(job_id, {"status": "downloading"})
        download_file(s3_key, local_path)

        duration = get_audio_duration(local_path)
        model = get_model(model_name)

        # =========================
        # 转录
        # =========================
        if duration > 600:
            set_job(job_id, {"status": "splitting"})

            chunks = split_audio(local_path, chunk_duration=60)
            all_segments = []

            for i, chunk in enumerate(chunks):
                set_job(job_id, {"status": f"chunk_{i}"})
                result = model.transcribe(chunk)
                all_segments.append(result["segments"])

            segments = merge_segments(all_segments)

        else:
            set_job(job_id, {"status": "transcribing"})
            result = model.transcribe(local_path)
            segments = result["segments"]

        # =========================
        # 翻译
        # =========================
        if translate:
            if target_language and target_language.lower() != "en":
                set_job(job_id, {"status": "translating_openai"})
                segments = translate_segments_openai(segments, target_language)
            else:
                set_job(job_id, {"status": "translating"})
                result = model.transcribe(local_path, task="translate")
                segments = result["segments"]

        # =========================
        # 生成字幕
        # =========================
        srt = generate_srt(segments)

        output_local = f"/tmp/{job_id}.srt"
        with open(output_local, "w") as f:
            f.write(srt)

        # 上传 S3
        output_key = f"output/{job_id}.srt"
        upload_file(output_local, output_key)

        # 更新状态
        set_job(job_id, {
            "status": "ready",
            "output": output_key
        })

        return {"status": "ready", "output": output_key}

    finally:
        release_slot(model_name)

        if os.path.exists(local_path):
            os.remove(local_path)
