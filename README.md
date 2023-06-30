# mma-project
Multimedia Analytics project repository

To run the code, set the environment variables to the correct file paths:
```
For accessing css and js files:
   ASSETS_PATH = ./assets
For accessing the pickle file with the preprocessed data:
   ADS_PATH = ./data/final.pkl
For accessing the precomputed image embeddings for the entire funda dataset:
   EMBEDDINGS_PATH = ./data/full_embeddings.pkl (download from drive: https://drive.google.com/file/d/1tOevR0Mlte-oEYLmmDnZ_v1K7uJMsVj0/view?usp=sharing)
For accessing the image files in the dataset:
   IMAGES_PATH = ./images (download and unpack: https://amsuni-my.sharepoint.com/:f:/g/personal/t_j_vansonsbeek_uva_nl/EutCd4e9a9hEpEFZUzU7tiAB1be1dou-bRA2gl_dSFhvsQ)
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

Run code from src folder:
```
python3 app.py
```

