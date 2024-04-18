﻿# Final Model (Trained on FaSolLa)
https://drive.google.com/file/d/1EZS7JcPvDdNSfBWdQs0B705uzMJTLkTd/view?usp=sharing

# Test images from FaSolLa
https://drive.google.com/file/d/1948a6Cnv5U_6Y9VmYdDHQFKTeovrx5uM/view?usp=sharing

# Corpus
Folder hierarchy of the corpus user in the training

Dataset/&emsp;&emsp;&emsp;# Root directory of the corpus  
├── images/&emsp;&emsp;# Image (.png) of the partition segments  
├── labels_length/&emsp;# Rhythm labels (.semantic). Generated by following the label generation pipeline  
├── labels_note/&emsp;# Pitch labels (.semantic). Generated by following the label generation pipeline  
├── copy_test.ps1&emsp;# Script to create a test folder containing the images listed in the test.txt file  
├── fix_labels.ps1&emsp;# Script to normalize label file names  
├── train.txt&emsp;&emsp;# List of the training images  
├── valid.txt&emsp;&emsp;# List of the validation images  
└── test.txt&emsp;&emsp;# List of the test images  
        
# Label Generation

## Requirements
1. Install MuseScore 3 : https://musescore.org/en/download (MuseScore 4 won't work, see the links for older version at the bottom of the page)
2. Copy folder from the folder (label_gen/musescore_plugins) in MuseScore's Plugin folder : https://musescore.org/en/handbook/3/plugins
3. Make sure the plugins are activated in the menu Plugins/Plugin Manager of MuseScore

## Pipeline
Information : "Batch Convert Orig" and "Batch Convert Resize Height" can be run from the Plugins menu of MuseScore.
Warning     : Running "Batch Convert" on a folder will delete the original files and replace them with the output file. Make sure to do a backup if you want to keep the originals.

0. (Optionnal) If your starting files are .musicxml : Run "Batch Convert Orig" in MuseScore on all .musicxml files to .mscz
1. Run "Batch Convert Resize Height" in MuseScore on all .mscz files to .musicxml
2. Run `removecredits.py -input "MusicXMLFolder"` on the generated .musicxml files
3. Run "Batch Convert Orig" in MuseScore on the cleaned .musicxml files to .mscz
4. Run "Batch Convert Orig" in MuseScore on the new .mscz to .musicxml and .png
5. Run `removetitleimgs.py -input "ImageFolder"` on the images (Essentially removes the first sample of each partition to remove the title image)
6. Run `genlabels.py -input "MusicXMLFolder" -output "LabelsFolder" -voc_p "PathToProject\experiment_code\vocab\rnn_pitch.txt" -voc_r "PathToProject\experiment_code\vocab\rnn_rhythm.txt"` to generate labels for the .musicxml files
7. Run `removesparsesamples.py -input "ImageAndAnySemanticFolder"` to remove samples containing only rests
8. Run `removenolabeldata.py -labels "AnySemanticFolder" -imgs "ImageFolder"` to samples that don't have labels
9. Create the corpus using the created images and finialized labels
10. Run the script fix_labels.ps1 (Which needs to be at the root of the corpus) to fix label names.

# Model Training

## Requirements
1. Have a valid Corpus

## Pipeline
The model will be in a folder named model. A model is saved for each epoch.

1. Run `train_multi.py -voc_p "PathToProject\experiment_code\vocab\rnn_pitch.txt" -voc_r "PathToProject\experiment_code\vocab\rnn_rhythm.txt" -corpus "DatasetFolder"`

# Prediction

## Requirements
1. Have a valid model

## Pipeline

1. Generate the Pitch prediction with `predict_multi.py -p -images "TestImagesFolder" -model "ModelsFolder/model_name.pt" -voc_p "PathToProject\experiment_code\vocab\rnn_pitch.txt" -voc_r "PathToProject\experiment_code\vocab\rnn_rhythm.txt" -out "PitchPredictionFolder"`
2. Generate the Rhythm prediction with `predict_multi.py -images "TestImagesFolder" -model "ModelsFolder/model_name.pt" -voc_p "PathToProject\experiment_code\vocab\rnn_pitch.txt" -voc_r "PathToProject\experiment_code\vocab\rnn_rhythm.txt" -out "RhythmPredictionFolder"`

# Music XML Generation from labels

## Requirements
1. Have predictions for rhythms and pitch

# Pipeline

1. Generate the Music XML with `generate_music_xml.py -notes "PathToPredictionFolder\pitch" -rhythms "PathToPredictionFolder\rhythm" -output "PathToMusicXMLFolder"`