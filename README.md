#
# README
# 
### Author : Peter HIRT
### date : July 5 2017

This is the folder with the starterkit to do UNET training and inference on either artificial data or on real medical images.

# Installation

all work was done with Python 3.5
then Tensorflow v1.1 and opencv v3.1
the rest of the packages is best installed with anaconda or pip. No comprehensive list of packages is given here.
Also, some imports may actually not be used, hence proceed with some trials.

## File Description

FCN_jul4.py 			:  main file to launch the training.
				
ModelLibrary.py 		:  defines the UNET model with the given parameters

BatchDatsetReader.py 	:  reader for medical images prepared in the data directory (here dataJul4)

DatasetReader.py   		:  reader for generator images

TensorflowUtils.py      :  generic utilities

## Notebook Description

Inference Greyscale FCN-UNET.ipynb 	:   launches the inference on samples and visualises result

ROI processing-Jul4-stripped.ipynb  :   creates the data structure for medical images based on DICOM files and text files 
                                        for ROIs. When done, zip the content and copy it into the data... folder

HUG_data_reader.ipynb    			:    visualisation of data from the 2 Readers
										 ATTENTION  : THIS NOTEBOOK IS FOR THE KERAS BASED FCN



## Folder Description

data....				:  folder for the medical image data. It expects to see at least a .zip with the images in .jpg
  						   format and the labels in .png format

logs  					:  model checkpoint folder 
						   events folder for TensorBoard

tf_unet  				:  folder with helper utilities for UNET model 