from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import subprocess
import os

app = FastAPI()


@app.post("/visualize_brain")
async def visualize_brain(file: UploadFile = File(...), myNewId: str = Form(...)):
    try:
        upload_dir = os.path.join("data", "uploads", myNewId)
        os.makedirs(upload_dir, exist_ok=True)

        # Save the uploaded file to the specified location
        upload_path = os.path.join(upload_dir, file.filename)
        with open(upload_path, "wb") as f:
            f.write(file.file.read())

        # Run brain_visualizer.py with myNewId as an argument
        subprocess.run(["python", "brain_visualizer.py",
                       "--file", upload_path, "--newId", myNewId])

        return JSONResponse(content={"message": "Brain visualization triggered."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
