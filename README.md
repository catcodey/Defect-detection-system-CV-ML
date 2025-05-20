# Defect-detection-system-CV-ML

**RUN THE "complete code without button and scanner.py file".**  

Only this file correctly displays output in flask webpage. The other python file "10-12-2024..." still doesn't work as the paths are yet to be properly modified.

This is a ai based component inspection system built using yolov8 model to identify defects on turbo chargers in real time as they come down the assembly line. A variety of techniques have been put to use for performing various tasks. 

The following defects have been identified and inspected:  
- Presence or absence of
   - Eclips
   - End link Crimp(ELC)
   - NBRR (Noise Baffle and Retainer Ring)
   - Dataplate
   - Drivescrews
- Object detection used for the above defects
- Locknut bolt loose/tight --> keypoint detection

- ALL TRAINED MODELS ARE PRESENT IN THIS KAGGLE LINK SINCE THE FILE SIZE IS TOO BIG TO UPLOAD IN GITHUB [Trained Models Kaggle Link](https://www.kaggle.com/models/bhavnab/defect-detection-training-models-and-wts)
- captured images new folder is the save_folder. This is where the outputs are saved.
- Here's another read me file which explains the file structure and how to run it [Google doc read me](https://docs.google.com/document/d/1tCC1y46SGV-2O-KF5sP12R3Th_FADdJ1Jfke5F4nMNA/edit?tab=t.0)
