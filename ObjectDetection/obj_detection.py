#!/usr/bin/env python
# coding: utf-8

#Baseline code is Tensorflow model object detection API 

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO

import argparse
from datetime import datetime


from PIL import Image
import imageio


# Change this to wherever you cloned the tensorflow models repo
# Downloaded the tensorflow code from:
# https://github.com/tensorflow/models
RESEARCH_PATH = r'/home/trushk/ml/tf_models/models/research'
MODELS_PATH = r'/home/trushk/ml/tf_models/models/research/object_detection'

sys.path.append(RESEARCH_PATH)
sys.path.append(MODELS_PATH)

from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')


# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.


# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = r'./models/ssd_mobilenet_v1_coco/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '%s/data/mscoco_label_map.pbtxt' % MODELS_PATH

# ## Download Model

"""
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.


"""
#Tensorflow Graph
detection_graph = tf.Graph()

with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


# ## Helper code




def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection




# Size, in inches, of the output images.
IMAGE_SIZE = (8, 8)

img_width = 1024
img_height= 512


def run_inference_for_single_image(image, graph):
    
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


#Argument parser
parser = argparse.ArgumentParser()

#Optional arg to specify image directory for inference
parser.add_argument("-i","--img_dir",help="Path to image dir")
#Optional arg to specify video for inference
parser.add_argument("-v","--video",help="Video to test")

args = parser.parse_args()

#Handle images
if args.img_dir :
  
    print ("Processing images in %s" % args.img_dir)
    
    TEST_IMAGE_PATHS = [ os.path.join(args.img_dir, 'image{}.jpg'.format(i)) for i in range(1, 4) ]

    for image_path in TEST_IMAGE_PATHS:
        
        image = Image.open(image_path)
 
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
  
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np, detection_graph)
  
        # Visualization of the results of a detection.
  
        vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=8)
  
  
        img = Image.fromarray(image_np)
        new_img = img.resize((img_width,img_height))
        new_img.show()



#Handle a video
#sample videos can be downloaded here:
#   http://bvisionweb1.cs.unc.edu/ilsvrc2015/download-videos-3j16.php#vid

if args.video:
    
    input_video = args.video

    print ("Processing %s.mp4" % input_video)
    
    video_reader = imageio.get_reader('%s.mp4' % input_video)
    video_writer = imageio.get_writer('%s_annotated.mp4' % input_video, fps=10)


    # loop through and process each frame
    t0 = datetime.now()
    n_frames = 0
    for frame in video_reader:
      # rename for convenience
      image_np = frame
      n_frames += 1

      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      
      output_dict = run_inference_for_single_image(image_np, detection_graph)

      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      
      # instead of plotting image, we write the frame to video
      video_writer.append_data(image_np)

    fps = n_frames / (datetime.now() - t0).total_seconds()
    print("Frames processed: %s, Speed: %s fps" % (n_frames, fps))

    print ("Generated %s_annotated.mp4" % input_video)
    
    # clean up
    video_writer.close()

