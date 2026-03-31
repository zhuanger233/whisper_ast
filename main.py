from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
import shutil
import uuid
import os

from tasks import transcribe_task
from utils import get_audio_duration
from scheduler import choose_model
from redis_client import set_job, get_job
from s3_client import upload_file, generate_presigned_url

app = FastAPI()


class JobCreateRequest(BaseModel):
    filename: str = Field(..., min_length=1)
    target_language: str = Field(..., min_length=2, max_length=10)
    translate: bool = False


@app.post("/jobs")
async def create_job(payload: JobCreateRequest):
    job_id = f"job_{uuid.uuid4().hex}"
    set_job(job_id, {
        "status": "created",
        "filename": payload.filename,
        "target_language": payload.target_language,
        "translate": payload.translate
    })
    return {
        "job_id": job_id,
        "filename": payload.filename,
        "target_language": payload.target_language,
        "translate": payload.translate
    }


@app.post("/jobs/{job_id}/upload")
async def upload(job_id: str, file: UploadFile = File(...)):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    expected_filename = job.get("filename")
    if expected_filename and file.filename != expected_filename:
        raise HTTPException(status_code=400, detail="filename mismatch")

    local_path = f"/tmp/{job_id}_{file.filename}"

    # 临时保存
    with open(local_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    duration = get_audio_duration(local_path)
    model = choose_model(duration)

    # 上传到 S3
    s3_key = f"input/{job_id}/{file.filename}"
    upload_file(local_path, s3_key)

    # 删除本地文件（可选）
    os.remove(local_path)

    # 提交任务
    translate = bool(job.get("translate", False))
    target_language = job.get("target_language")
    task = transcribe_task.delay(
        job_id,
        s3_key,
        model,
        translate=translate,
        target_language=target_language
    )

    set_job(job_id, {
        "status": "queued",
        "task_id": task.id,
        "model": model,
        "input": s3_key
    })

    return {
        "job_id": job_id,
        "model": model,
        "target_language": target_language,
        "translate": translate
    }


@app.get("/jobs/{job_id}/status")
async def status(job_id: str):
    job = get_job(job_id)
    if not job:
        return {"error": "job not found"}

    response = {"status": job.get("status")}

    # 如果任务完成，附加可下载 URL
    if job.get("status") == "ready" and "output" in job:
        response["download_url"] = generate_presigned_url(job["output"])

    return response
