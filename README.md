# mce2018
An implementation of an open-set speaker recognition system for the 1st Multi-target speaker detection and identification Challenge Evaluation (MCE 2018, http://mce2018.org/).

The baseline system of the challenge can be found here: https://github.com/swshon/multi-speakerID. The helper functions used in the implementation of our system are taken from there.

After downloading the data (https://www.kaggle.com/kagglesre/blacklist-speakers-dataset) and installing all required dependencies, you can simply run the script evaluate_system.py which will do everything automatically.

The PLDA script that is used is an adaptation of the MATLAB files of the fastPLDA toolkit (see https://sites.google.com/site/fastplda/) to Python.

Note, that it is very unlikely to obtain exactly the same performance as presented in the paper because of nondeterministic behavior. However, the EERs should be about the same.

When finding this code helpful, or reusing parts of it, a citation would be appreciated:

@article{wilkinghoff2019openspeaker,
  title={Training an Open-Set Speaker Recognition System Without Using Non-Blacklist Speakers},
  author={Wilkinghoff, Kevin},
  journal={Preprint (submitted)},
  year={2019}
}
