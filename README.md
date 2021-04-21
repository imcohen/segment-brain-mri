# Brain MRI Segmentation

Welcome to this tutorial !

This repository implements brain MRI segmentation methods from Kaggle dataset :
- Minimal-path extraction using Fast-Marching algorithm (tutorial 1)
- Deep-learning UNet model to be trained (tutorial 2)

Please, first clone the repo.

Then, download the dataset (~2 GB) from https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation and unzip it into the 'data' directory.

The code is written with Python 3.8, and uses a PyTorch implementation of UNet (https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/).

## Requirements

### Using Conda

Having Conda installed on your local machine, you can create a new Conda environment for the project using the file [environment.yml](environment.yml).

Run the following lines from the project root directory to create the environment and activate it :

```
conda env create -f environment.yml
conda activate segment-brain-mri
```

Then, to create a Jupyter kernel associated to your Conda environment, please run :

```
python -m ipykernel install --user --name=segment-brain-mri
```

When you will launch Jupyter Notebook, select the kernel 'segment-brain-mri'.

### Using Docker

You must have [Docker](https://www.docker.com/) installed on your local machine.

From the project root directory, run the following docker command to build the Docker image 'segment-brain-mri' from the [Dockerfile](Dockerfile) :
```
docker build . -t segment-brain-mri
```

<ins>Note</ins>: The size of the Docker image is about 4Go. If you encounter a problem of memory, you probably need to increase your disk image size. You can modify it by going to 'Docker > Preferences > Disk image size'.

When the build is finished, you can start a container based on your docker image 'segment-brain-mri' :
```
docker run -it -v $PWD:/home/segment-brain-mri -p 5000:5000 segment-brain-mri bash
```

To start a Jupyter notebook session, run the following command :
```
jupyter notebook --port=5000 --ip=0.0.0.0 --allow-root
```
Copy/paste the output URL with port 5000 to your browser.


## Notebooks

To have a quick overview of the dataset, open the Jupyter notebook
[dataset_overview.ipynb](notebooks/dataset_overview.ipynb)
 
For tutorial 1, open the Jupyter notebook [fast_marching_segmentation.ipynb](notebooks/fast_marching_segmentation.ipynb).
For tutorial 2, see next section (Colab).

## Colab notebook

To use free GPU computing to train your deep-learning model (tutorial 2), use Google Colab!

In Colab (or you can first upload it to your Google Drive), open the notebook [deep_learning_segmentation.ipynb](notebooks/deep_learning_segmentation.ipynb), and follow the instructions.