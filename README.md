# Improved MTCNN TF2 Training

This repository contains an improved version of MTCNN training using TensorFlow 2. It is based on [AITTSMD/MTCNN-Tensorflow](https://edgeservices.bing.com/edgesvc/%5E1%5E), which reproduces the original MTCNN paper. The main enhancements are:

* Simpler and clearer code structure
* Easy to execute
* Compatible with TensorFlow 2.11 or lower
* Support for CUDA GPU acceleration
* Python 3.10 compatible

All you need is the training dataset from [Wider Face Training Dataset](http://shuoyang1213.me/WIDERFACE/ "Face") and [CNN Facepoint Training Dataset](http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm "Landmarks"), then put it in the dataset folder.

 **Key Points** :

* `gen_hard_bbox_pnet.py` is used to generate 12x12 training images for PNet
* `gen_hard_bbox_rnet_onet.py` is used to generate 24x24 training images for RNet, and 48x48 training images for ONet
* `gen_landmark_aug.py` is used to generate 12x12, 24x24, and 48x48 facial landmark images for all the networks
* `training.py` is used to train PNet, RNet, and ONet
* `gen_tfrecords.py` is used to save training data in tfrecord format for easy training

Training Steps：

* PNet
  1. Run `python gen_hard_bbox_pnet.py`
  2. Run `python gen_landmark_aug.py --stage pnet`
  3. Run `python gen_tfrecords.py --stage pnet`
  4. Run `python train.py --stage pnet`
* RNet
  5. Run `python gen_hard_bbox_rnet_onet.py --stage rnet`
  5. Run `python gen_landmark_aug.py --stage rnet`
  5. Run `python gen_tfrecords.py --stage rnet`
  5. Run `python train.py --stage rnet`
* ONet
  9. Run `python gen_hard_bbox_rnet_onet.py --stage onet`
  9. Run `python gen_landmark_aug.py --stage onet`
  9. Run `python gen_tfrecords.py --stage onet`
  9. Run `python train.py --stage onet`

And you're done! Enjoy the results! :)
