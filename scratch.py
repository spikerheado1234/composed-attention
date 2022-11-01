## This is a scratch pad for playing around with things. ##

"""
datasests library is downloaded with the following command:
`pip install datasets`
"""
import datasets
from pathlib import Path

target_path='/home/ag82/scratch/c4'

datasets.config.DOWNLOADED_DATASETS_PATH = Path(target_path)

datasets.load_dataset('c4', 'en', ignore_verifications=True)

