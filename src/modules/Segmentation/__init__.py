"""Model-based note segmentation.

A small neural network predicts, frame by frame on the separated vocal, where
notes start and which passages are charted at all. Lyrics are then placed onto
the predicted notes. The network is trained with ``tools/train_segmentation.py``
on an existing UltraStar song library; no model is shipped with UltraSinger.
"""
