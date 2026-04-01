import os
import json
import re
import logging
import whisper
import boto3
from botocore.exceptions import ClientError
from worker import celery
from scheduler import acquire_slot, release_slot
from utils import generate_srt, split_audio, merge_segments, get_audio_duration
from redis_client import set_job
from s3_client import download_file, upload_file

models = {}
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


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


def _get_bedrock_client():
    region = os.getenv("BEDROCK_REGION") or os.getenv("AWS_REGION") or "us-west-2"
    return boto3.client("bedrock-runtime", region_name=region)


def translate_segments_bedrock(segments, target_language: str):
    client = _get_bedrock_client()
    translated = []
    batch_size = 50

    logger.info("bedrock_translate_start target_language=%s segments=%d", target_language, len(segments))

    model_id = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-3-7-sonnet-20250219-v1:0")
    anthropic_version = os.getenv("BEDROCK_ANTHROPIC_VERSION", "bedrock-2023-05-31")
    temperature = float(os.getenv("BEDROCK_TEMPERATURE", "0.2"))
    top_p = float(os.getenv("BEDROCK_TOP_P", "0.8"))
    top_k = int(os.getenv("BEDROCK_TOP_K", "80"))
    max_tokens = int(os.getenv("BEDROCK_MAX_TOKENS", "2048"))

    for i in range(0, len(segments), batch_size):
        batch = segments[i:i + batch_size]
        items = [{"text": s["text"]} for s in batch]
        input_json = json.dumps(items, ensure_ascii=False)

        try:
            prompt = (
                "You are a subtitle translation engine. "
                "Translate each item to the target language. "
                "Return ONLY a JSON array of strings in the same order, with no extra text.\n"
                f"Target language: {target_language}\n"
                f"Input JSON:\n{input_json}"
            )
            native_request = {
                "anthropic_version": anthropic_version,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    }
                ],
            }
            response = client.invoke_model(
                modelId=model_id,
                body=json.dumps(native_request),
                contentType="application/json",
                accept="application/json",
            )
        except (ClientError, Exception) as e:
            logger.exception("bedrock_translate_error batch_start=%d batch_size=%d", i, len(batch))
            raise e

        model_response = json.loads(response["body"].read())
        output_text = model_response["content"][0]["text"]
        snippet = (output_text[:200] + "...") if output_text and len(output_text) > 200 else output_text
        logger.info(
            "bedrock_translate_response batch_start=%d batch_size=%d output_text_len=%s snippet=%s",
            i, len(batch), (len(output_text) if output_text else "None"), snippet
        )

        if not output_text:
            raise ValueError("Empty Bedrock response output_text")

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
        logger.info(
            "job_start job_id=%s translate=%s target_language=%s duration=%.2f",
            job_id, translate, target_language, duration
        )
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
                set_job(job_id, {"status": "translating_bedrock"})
                segments = translate_segments_bedrock(segments, target_language)
            else:
                if not target_language:
                    logger.info("translate_fallback_whisper_no_target_language job_id=%s", job_id)
                else:
                    logger.info("translate_whisper_en job_id=%s target_language=%s", job_id, target_language)
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
