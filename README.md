# Email Spam Detector

### Current Dataset Setup

- Training archive in repo: `dataset/train.zip`
- Test archive in repo: `dataset/test.zip`

After extracting the archives locally, the code expects:
- training CSV at `dataset/train/processed_data.csv`
- test email folders under `dataset/test/`

The project now trains on the processed TREC 2007 CSV after extraction.
This replaced the previous Enron CSV training setup after we found serious issues in the
original training file.

### Usage

- pip install -r requirements.txt
- python3 main.py

If `dataset/train/` or `dataset/test/` do not exist, `main.py` will try to extract
`dataset/train.zip` and `dataset/test.zip` automatically before loading the data.
