# Image Colorization
This directory contains the code needed to conduct Image Colorization. All code files were written in Python. 

## Directory Layout

```text
Image-Colorization/
|-- README.MD
|-- requirements.txt
|-- input/
|   `-- *.jpg                    # Images selected for colorization
|-- places365/
|   |-- places365_val.txt        # Places365 validation file list
|   `-- val_256/
|       `-- *.jpg                # Places365 validation images
`-- code/
    |-- download_data.py         # Downloads the Places365 data
    |-- convert_dataset.py       # Converts the dataset for training
    |-- colorNet.py              # Model architecture
    |-- train.py                 # Training script
    |-- evaluate.py              # Evaluation script
    |-- color_image.py           # Applies a trained model to an image
    |-- logs/
    |   `-- *.txt                # Training and evaluation logs
    |-- outputs/
    |   |-- *.pt                 # Saved model checkpoints
    |   `-- *_samples_epoch*.png # Training sample grids
    `-- colorized/
        `-- *_color.png          # Colorized output images
        `-- *_compare.png        # Side-by-side comparison images
```

## Running the Code
### Installing Requirements
Ensure you install the necessary requirements in **requirements.txt** using the following command: 
```
python -m pip install -r requirements.txt
```

### Downloading the Dataset
To download Places365, the dataset I am using to train and test the models, run the **download_data.py** file using the following command (ensure you are in the **code** folder):
```
python download_data.py
```

### Model Training
To train the models use the following commands:
```
New-Item -ItemType Directory -Force -Path logs | Out-Null
python -u train.py --mode 1 --num_epochs 20 2>&1 | Tee-Object -FilePath logs/log_mode1.txt
python -u train.py --mode 2 --num_epochs 20 2>&1 | Tee-Object -FilePath logs/log_mode2.txt
python -u train.py --mode 3 --chroma_weight 0.5 --num_epochs 20 2>&1 | Tee-Object -FilePath logs/log_mode3_w0.5.txt
python -u train.py --mode 3 --chroma_weight 1.0 --num_epochs 20 2>&1 | Tee-Object -FilePath logs/log_mode3_w1.0.txt
```
These commands will run the **train.py** file at different modes and save the console output in the **logs** folder for later reference.

It is important to note that training the models may take a while depending on your computer's hardware. Additionally each run will get slightly different weights for each model. 

### Evaluating the Models
To determine how well each model did, run the **evaluation.py** file:
```
python evaluate.py --mode 1
python evaluate.py --mode 2
python evaluate.py --mode 3 --weights outputs/colornet_mode3_w0.5_last.pt
python evaluate.py --mode 3 --weights outputs/colornet_mode3_w1.0_last.pt
```

### Using the Models
To run the models use the following commands:

For a folder:
```
python color_image.py --mode 1 --input ../input
python color_image.py --mode 2 --input ../input
python color_image.py --mode 3 --weights outputs/colornet_mode3_w0.5_last.pt --input ../inputs
python color_image.py --mode 3 --weights outputs/colornet_mode3_w1.0_last.pt --input ../inputs
```

For a single image: 
```
python color_image.py --mode <mode-number> --input ../input/<image-name>.jpg
```