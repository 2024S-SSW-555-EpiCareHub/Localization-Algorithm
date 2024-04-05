from fastapi import FastAPI
import subprocess

app = FastAPI()


@app.get("/visualize_brain")
def visualize_brain():
    # Run the separate Python script for brain visualization
    subprocess.run(["python", "brain_visualizer.py"])
    return {"message": "Brain visualization triggered."}
