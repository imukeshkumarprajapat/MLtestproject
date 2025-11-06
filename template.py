import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logging.info("Starting the project setup...")
#यह लाइन Python में लॉगिंग सिस्टम को सेट करती है ताकि प्रोग्राम के दौरान कौन-कौन से मैसेज दिखाए जाएं, यह तय किया जा सके
project_name = "ml_first_project"
list_of_files = [

    ".github/workflows/.gitkeep",
    #GitHub ka use hm deployment karte time action workflows me likhte h .gitkeep file se folder ko track
    #  krne k liye use krte h halanki .gitkeep file khud me khali hoti h (dummy file )  kyoki 
    # workflow folder me kuch files nahi hoti h to git us(empty) folder ko track nahi krta isliye .gitkeep file use krte h
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_monitering.py",   
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/training_pipeline.py",
    f"src/{project_name}/pipleines/prediction_pipeline.py",
    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/utils.py",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",


]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory: {filedir} for file: {filename}")
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Created empty file: {filepath}")
    else:
        logging.info(f"File already exists: {filepath}, skipping creation.")