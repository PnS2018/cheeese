README FOR CHEEESE
==================

Description
-------------

Analysis of selfies to determine between good and bad!

How to take a good selfie?


Python Files
--------------
### main.py ###
In here, our model is trained...
 Currently the most siple approach achieves the best results (validation accuracy of ~ 69.5%), with 32 by 32 sized images and color channel set to 1. The other models were not tested for long, but most of them also reach close to 70% validation accuracy, depending on image size and color / black and white.

### utils.py ###
In here, a few helpful functions are defined...

### cam.py ###
In here, the interface for the camera is implemented...
 The camera captures the frames, which are then predicted (good/bad) by the loaded (trained) model. The accuracy (of the prediciton) is printed out to the console (under 0.5 -> bad/ over 0.5 -> good). When the camera is quit (using 'q'), the best image will be displayed and you will be asked, if you want to save the image.


Data Set
----------
You can find the data set we used under https://polybox.ethz.ch/index.php/f/962785156. There you will find the original data set, as described below, and also the pre sized numpy arrays we used for training.



"Selfie dataset contains 46,836 selfie images annotated with 36 different attributes divided into several categories as follows. Gender: is female. Age: baby, child, teenager, youth, middle age, senior. Race: white, black, asian. Face shape: oval, round, heart. Facial gestures: smiling, frowning, mouth open, tongue out, duck face. Hair color: black, blond, brown, red. Hair shape: curly, straight, braid. Accessories: glasses, sunglasses, lipstick, hat, earphone. Misc.: showing cellphone, using mirror, having braces, partial face. Lighting condition: harsh, dim." (Source: http://crcv.ucf.edu/data/Selfie/)


Since we make use of the selfie data set we cite the following work:

@inproceedings
{
    kalayeh2015selfie,
    title={How to Take a Good Selfie?},
    author={Kalayeh, Mahdi M and Seifu, Misrak and LaLanne, Wesna and Shah, Mubarak},
    booktitle={Proceedings of the 23rd Annual ACM Conference on Multimedia Conference},
    pages={923--926},
    year={2015},
    organization={ACM}
}
