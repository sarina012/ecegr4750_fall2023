import numpy as np
import pandas as pd
from PIL import Image

# define function to standardize numerical features of dataframe
def standardize_numeric(series: pd.Series, use_log: bool = False) -> pd.Series:
    if use_log:
        series = np.log(series)
    return (series - np.mean(series))/np.std(series)

# define function to convert PIL images into correct format for exporting and model training
# Images are resized down to 64 x 64 pixels so that image processing does not exceed memory
# This further reduces time for image processing
def process_images(image_filenames: pd.Series) -> np.ndarray:
    
    images = []

    for image_path in image_filenames:
        im = Image.open('data/images/' + image_path)
        im = im.resize((64, 64))
        images.append(np.moveaxis(np.asarray(im), 2, 0))
    
    return np.asarray(images)