# mma-project
Multimedia Analytics project repository

To run the code, set the environment variables to the correct file paths:
```
For accessing css and js files:
   ASSETS_PATH = ./assets
For accessing the pickle file with the preprocessed data:
   ADS_PATH = ./data/final.pkl
For accessing the precomputed image embeddings for the entire funda dataset:
   EMBEDDINGS_PATH = ... (download from drive)
For accessing the image files in the dataset:
   IMAGES_PATH = ... (university supplied)
```

Install requirements:
```
pip install -r requirements.txt
```

Run code:
```
python3 src/app.py
```

