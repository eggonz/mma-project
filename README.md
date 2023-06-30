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

If you want to use GPT, provide a valid openai API key as an environment variable:
```
For using GPT:
   OPENAI_API_KEY = ...
```
and comment line 28 in app.py!

Use `.env` for your convenience.

Install requirements:
```
pip install -r requirements.txt
```

Run code:
```
python3 src/app.py
```

