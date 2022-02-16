# Pain Detection

## Pre-requestiue:

Download the weight file and place it in the right folder:

1. Download s3fd-619a316812.pth:
https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth and place it under 'face_alignment\detection\sfd' folder.

2. Download this weight file:
        For 2D: https://www.adrianbulat.com/downloads/python-fan/2DFAN-4.pth
        For 3D: https://www.adrianbulat.com/downloads/python-fan/3DFAN-4.pth
   And place it under face_alignment folder.

A minimal repository (including pre-trained models) to demo the pain detection model proposed in the paper titled [Unobtrusive Pain Monitoring in Older Adults with Dementia
using Pairwise and Contrastive Training](https://ieeexplore.ieee.org/document/9298886). 

After installing the requirements, you should be able to run `test_image.py` or `test_video.py`, and it should print out the pain score (PSPI) for the frames in the `example_frames` folder.

## Two pretrained models are included:

One was trained on the UNBC-McMaster _Shoulder Pain Expression Archive_ dataset and the University of Regina's _Pain in Severe Dementia_ dataset.
And another checkpoint that was trained on the UNBC-McMaster dataset **only**. In both cases, UNBC subjects 66, 80, 97, 108, and 121 were excluded from training.

Currently [Face Alignment Network (and S3FD)](https://github.com/1adrianb/face-alignment) are used to detect and align faces.
These could be swapped for faster non-deep learning methods to improve performance.
