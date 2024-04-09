from fastapi import FastAPI, File, UploadFile, Form, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os
import uuid
import json
import platform


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:5173",  # Assuming your frontend runs on port 3000
    # Add other allowed origins as needed
]

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

with open('config.json', 'r') as file:
    config = json.load(file)

os_name = platform.system()

# Check if the OS is macOS or Windows
if os_name == 'Darwin':
    base_Path = config['mac_path']
    print("MacOS Detected", base_Path)
elif os_name == 'Windows':
    base_Path = config['windows_path']
else:
    raise Exception("Unsupported operating system.")
    


@app.post("/visualize_brain")
async def visualize_brain(file: UploadFile = File(...), patientId: str = Form(...)):
    try:

        uploadId = str(uuid.uuid4())
        upload_dir = os.path.join(base_Path, uploadId)
        os.makedirs(upload_dir, exist_ok=True)

        # Save the uploaded file to the specified location
        upload_path = os.path.join(upload_dir, file.filename)
        with open(upload_path, "wb") as f:
            f.write(file.file.read())

        subprocess.run(["python", "brain_visualizer.py","--basePath", base_Path,
                        "--file", upload_path, "--patientId", patientId, "--uploadId", uploadId, "--historic", str(False)])

        output_file = "output.json"
        if os.path.exists(output_file):
            with open(output_file, "r") as infile:
                data = json.load(infile)

            # Delete the output file after reading its contents
            os.remove(output_file)
        else:
            return JSONResponse(content={"message": "Brain visualization triggered."}, status_code=status.HTTP_200_OK)
        print(data)
        return JSONResponse(content={"message": "Brain visualization triggered.", "data": data}, status_code=status.HTTP_200_OK)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/visualize_brain_historic")
async def visualize_brain_historic(uploadId: str = Form(...)):

    try:
        # Save the uploaded file to the specified location
        upload_dir = os.path.join(base_Path, uploadId)

        if not os.path.exists(upload_dir):
            return JSONResponse(content={"message": "No Visualization record found."})

        subprocess.run(["python", "brain_visualizer.py",
                       "--upload_dir", upload_dir, "--historic", str(True)])

        return JSONResponse(content={"message": "Brain visualization triggered."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
