import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from detection.MtcnnDetector import MtcnnDetector
from detection.fcn_detector import FcnDetector
from detection.detector import Detector
from training.mtcnn_model import P_Net, R_Net, O_Net

# Load the trained TensorFlow model
pnet_model_path = './tmp/model/pnet/pnet-30'
rnet_model_path = './tmp/model/rnet/rnet-30'
onet_model_path = './tmp/model/onet/onet-30'

# Load the models using TensorFlow
pnet_detector = FcnDetector(P_Net, pnet_model_path)
rnet_detector = Detector(R_Net, 24, 64, rnet_model_path)
onet_detector = Detector(O_Net, 48, 16, onet_model_path)

# Create an instance of MtcnnDetector
mtcnn_detector = MtcnnDetector(detectors=[pnet_detector, rnet_detector, onet_detector])

# TensorFlow Lite Converter for P-Net
converter_pnet = tf.compat.v1.lite.TFLiteConverter.from_session(sess=pnet_detector.sess, input_tensors=[pnet_detector.image_op], output_tensors=[pnet_detector.cls_prob, pnet_detector.bbox_pred])
converter_pnet.optimizations = [tf.lite.Optimize.DEFAULT]
converter_pnet.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model_pnet = converter_pnet.convert()

# Save the TensorFlow Lite model to a file for P-Net
with open('./tflite/pnet_model.tflite', 'wb') as f:
    f.write(tflite_model_pnet)

converter_rnet = tf.compat.v1.lite.TFLiteConverter.from_session(sess=rnet_detector.sess, input_tensors=[rnet_detector.image_op], output_tensors=[rnet_detector.cls_prob, rnet_detector.bbox_pred])
converter_rnet.optimizations = [tf.lite.Optimize.DEFAULT]
converter_rnet.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model_rnet = converter_rnet.convert()

with open('./tflite/rnet_model.tflite', 'wb') as f:
    f.write(tflite_model_rnet)

# Convert R_Net to TFLite
converter_onet = tf.compat.v1.lite.TFLiteConverter.from_session(sess=onet_detector.sess, input_tensors=[onet_detector.image_op], output_tensors=[onet_detector.cls_prob, onet_detector.bbox_pred, onet_detector.landmark_pred])
converter_onet.optimizations = [tf.lite.Optimize.DEFAULT]
converter_onet.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model_onet = converter_onet.convert()

# Save the TFLite model to a file
with open('./tflite/onet_model.tflite', 'wb') as f:
    f.write(tflite_model_onet)