from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import subprocess
import os
import uuid


app = FastAPI()


@app.post("/visualize_brain")
async def visualize_brain(file: UploadFile = File(...), patientId: str = Form(...)):
    try:

        uploadId = str(uuid.uuid4())
        upload_dir = os.path.join("uploads", uploadId)
        os.makedirs(upload_dir, exist_ok=True)

        # Save the uploaded file to the specified location
        upload_path = os.path.join(upload_dir, file.filename)
        with open(upload_path, "wb") as f:
            f.write(file.file.read())

        subprocess.run(["python", "brain_visualizer.py",
                       "--file", upload_path, "--patientId", patientId, "--uploadId", uploadId, "--historic", str(False)])

        return JSONResponse(content={"message": "Brain visualization triggered."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/visualize_brain_historic")
async def visualize_brain_historic(uploadId: str = Form(...)):

    try:
        # Save the uploaded file to the specified location
        upload_dir = os.path.join("uploads", uploadId)

        if not os.path.exists(upload_dir):
            return JSONResponse(content={"message": "No Visualization record found."})

        subprocess.run(["python", "brain_visualizer.py",
                       "--upload_dir", upload_dir, "--historic", str(True)])

        return JSONResponse(content={"message": "Brain visualization triggered."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
