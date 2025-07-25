import os
import re
import sys
import socket
import signal
import logging
import logging.handlers
import configparser
import time

import numpy as np
import cv2
import json
import uuid
from pprint import pformat

# Add the nxai-utilities python utilities
if getattr(sys, "frozen", False):
  script_location = os.path.dirname(sys.executable)
elif __file__:
  script_location = os.path.dirname(__file__)
sys.path.append(os.path.join(script_location, "../nxai-utilities/python-utilities"))
import communication_utils

CONFIG_FILE = os.path.join(script_location, "..", "etc", "plugin.events.ini")

# Set up logging
LOG_FILE = os.path.join(script_location, "..", "etc", "plugin.events.log")

# Initialize plugin and logging, script makes use of INFO and DEBUG levels
logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s - %(levelname)s - example - %(message)s",
  filename=LOG_FILE,
  filemode="w",
)

# The name of the postprocessor.
# This is used to match the definition of the postprocessor with routing.
Postprocessor_Name = "Python-Pass-Event-Postprocessor"

# The socket this postprocessor will listen on.
# This is always given as the first argument when the process is started
# But it can be manually defined as well, as long as it is the same as the socket path in the runtime settings
Postprocessor_Socket_Path = "/tmp/python-pass-event-postprocessor.sock"


# Data Types
# 1:  //FLOAT
# 2:  //UINT8
# 3:  //INT8
# 4:  //UINT16
# 5:  //INT16
# 6:  //INT32
# 7:  //INT64
# 8:  //STRING
# 9:  //BOOL
# 11: //DOUBLE
# 12: //UINT32
# 13: //UINT64

def config():
  logger.info("Reading configuration from:" + CONFIG_FILE)

  try:
    configuration = configparser.ConfigParser()
    configuration.read(CONFIG_FILE)

    configured_log_level = configuration.get(
      "common", "debug_level", fallback="INFO"
    )
    set_log_level(configured_log_level)

    for section in configuration.sections():
      logger.info("config section: " + section)
      for key in configuration[section]:
        logger.info("config key: " + key + " = " + configuration[section][key])

  except Exception as e:
    logger.error(e, exc_info=True)

  logger.debug("Read configuration done")


def set_log_level(level):
  try:
    logger.setLevel(level)
  except Exception as e:
    logger.error(e, exc_info=True)


def signal_handler(sig, _):
  logger.info("Received interrupt signal: " + str(sig))
  sys.exit(0)


def main():
  # Start socket listener to receive messages from NXAI runtime
  server = communication_utils.startUnixSocketServer(Postprocessor_Socket_Path)

  # coefficient for translating to real coordinates because of the Nx three digits mantissa
  mantissa_coefficient = 1000

  H = {}

  known_points_cache = {}

  objects_cache = {}

  # Wait for messages in a loop
  while True:
    # Wait for input message from runtime
    logger.debug("Waiting for input message")

    try:
      input_message, connection = communication_utils.waitForSocketMessage(server)
      logger.debug("Received input message")
    except socket.timeout:
      # Request timed out. Continue waiting
      continue

    # Parse input message
    input_object = communication_utils.parseInferenceResults(input_message)

    device_id = input_object['DeviceID']

    lat_lon = {}

    known_points = {}

    logger.debug(input_object)

    # get image parameters
    width = input_object['Width']
    height = input_object['Height']

    timeout = float(input_object["ExternalProcessorSettings"]["externalprocessor.timeout"])

    for i in range(0, 10):

      key_string = f"externalprocessor.point{i}.figure"
      figure = json.loads(input_object["ExternalProcessorSettings"][key_string])

      logger.debug(figure)

      if figure.get('figure') is None:
        continue

      box_center = (
        ((figure['figure']['points'][1][0] - figure['figure']['points'][0][0]) / 2
         + figure['figure']['points'][0][0]) * width,
        ((figure['figure']['points'][1][1] - figure['figure']['points'][0][1]) / 2
         + figure['figure']['points'][0][1]) * height
      )

      if device_id not in known_points:
        known_points[device_id] = []
        known_points_cache[device_id] = []
        H[device_id] = None

      known_points[device_id].append({'pixel': box_center})

      key_string = f"externalprocessor.point{i}.latitude"
      latitude = float(input_object["ExternalProcessorSettings"][key_string])/mantissa_coefficient
      key_string = f"externalprocessor.point{i}.longitude"
      longitude = float(input_object["ExternalProcessorSettings"][key_string])/mantissa_coefficient

      known_points[device_id][i]['lat_lon'] = (latitude, longitude)

    logging.debug(known_points[device_id])
    logging.debug(len(known_points[device_id]))
    logging.debug(known_points[device_id] != known_points_cache[device_id])
    if known_points[device_id] != known_points_cache[device_id] and len(known_points[device_id]) >= 4:
      H[device_id] = compute_homography(known_points[device_id])
      known_points_cache[device_id] = known_points[device_id]

    logger.debug(H[device_id])
    if H[device_id] is not None:

      timestamp = float(input_object['Timestamp'])

      for class_name, bboxes in input_object["BBoxes_xyxy"].items():
        object_index = 0
        coordinate_counter = 0
        bbox_pixel = [0, 0, 0, 0]
        for bbox_coordinate in bboxes:
          bbox_pixel[coordinate_counter] = bbox_coordinate
          coordinate_counter += 1
          if coordinate_counter == 4:
            bbox_center = (
              ((bbox_pixel[2] - bbox_pixel[0]) / 2 + bbox_pixel[0]),
              ((bbox_pixel[3] - bbox_pixel[1]) / 2 + bbox_pixel[1])
            )
            lat, lon = apply_homography(H[device_id], bbox_center)

            input_object["ObjectsMetaData"][class_name]['AttributeKeys'][object_index].append("Latitude")
            input_object["ObjectsMetaData"][class_name]['AttributeKeys'][object_index].append("Longitude")
            input_object["ObjectsMetaData"][class_name]['AttributeValues'][object_index].append(str(lat))
            input_object["ObjectsMetaData"][class_name]['AttributeValues'][object_index].append(str(lon))

            # manage object cache
            object_id = input_object["ObjectsMetaData"][class_name]['ObjectIDs'][object_index]
            if object_id not in objects_cache:
              objects_cache[object_id] = {
                'type': {str(class_name): 1},
                'last_time_seen': time.time(),
                'object_id': str(uuid.UUID(bytes=object_id)),
                'geojson':[
                  {
                    'type': 'Feature',
                    'geometry': {
                      'type': 'Point',
                      'coordinates': [lat, lon]
                    },
                    'properties':{
                      'timestamp': timestamp
                    }
                  }
                ]
              }
            else:
              object_type = str(class_name)
              if object_type in objects_cache[object_id]['type']:
                objects_cache[object_id]['type'][object_type] += 1
              else:
                objects_cache[object_id]['type'][object_type] = 1

              objects_cache[object_id]['last_time_seen'] = time.time()
              objects_cache[object_id]['geojson'].append(
                {
                  'type': 'Feature',
                  'geometry': {
                    'type': 'Point',
                    'coordinates': [lat, lon]
                  },
                  'properties': {
                    'timestamp': timestamp
                  }
                }
              )

            coordinate_counter = 0
            object_index += 1

      for key in list(objects_cache.keys()):
        object_type = max(objects_cache[key]['type'], key=objects_cache[key]['type'].get)
        if time.time() - objects_cache[key]['last_time_seen'] >= timeout and object_type in ['car', 'truck', 'bus']:
          caption = "Object passed"
          description = json.dumps(objects_cache[key])


          del objects_cache[key]

          if "Events" not in input_object:
            input_object["Events"] = []
          input_object["Events"].append(
            {
              "ID": "nx.ai_manager_postprocessor.pass_event",
              "Caption": caption,
              "Description": description,
            }
          )



    logger.debug("Added event to output")

    # Write object back to string
    output_message = communication_utils.writeInferenceResults(input_object)

    # Send message back to runtime
    communication_utils.sendMessageOverConnection(connection, output_message)


def compute_homography(known_points):
  """Computes a homography transformation matrix using OpenCV."""

  logger.debug('computing homography')

  pixel_points = []
  real_world_points = []
  for known_point in known_points:
    pixel_points.append(known_point['pixel'])
    real_world_points.append(known_point['lat_lon'])

  pixel_points = np.array(pixel_points, dtype=np.float32)
  real_world_points = np.array(real_world_points, dtype=np.float32)

  H, _ = cv2.findHomography(pixel_points, real_world_points, method=cv2.RANSAC)

  return H


def apply_homography(H, pixel_coord):
  logger.debug('applying homography')
  """Applies a homography transformation to a pixel coordinate."""
  px, py = pixel_coord
  transformed = np.dot(H, np.array([px, py, 1]))
  X, Y = transformed[:2] / transformed[2]  # Normalize by Z
  return (X, Y)


if __name__ == "__main__":
  ## initialize the logger
  logger = logging.getLogger(__name__)

  ## read configuration file if it's available
  config()

  logger.info("Initializing car pass plugin")
  logger.debug("Input parameters: " + str(sys.argv))

  # Parse input arguments
  if len(sys.argv) > 1:
    Postprocessor_Socket_Path = sys.argv[1]
  # Handle interrupt signals
  signal.signal(signal.SIGINT, signal_handler)
  # Start program
  try:
    main()
  except Exception as e:
    logger.error(e, exc_info=True)
