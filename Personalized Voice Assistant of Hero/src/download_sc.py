# download_sc.py
import tensorflow_datasets as tfds

# this will download and prepare dataset into TFDS cache
ds_info = tfds.builder('speech_commands').download_and_prepare()
print("Done. Info:", ds_info)
