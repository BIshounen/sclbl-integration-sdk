import os
import re
import sys
import socket
import signal
import logging
import logging.handlers
import configparser
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
  # mantissa_coefficient = 1000
  #
  # H = None
  #
  # known_points_cache = {}

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

    lat_lon = {}

    known_points = {}

    # for setting_name, setting_value in input_object["ExternalProcessorSettings"].items():
    #   # get image parameters
    #   width = input_object['Width']
    #   height = input_object['Height']
    #
    #   if re.search(r"externalprocessor.point\d*.figure", setting_name):
    #     figure = json.loads(setting_value)
    #     logging.debug(figure)
    #
    #     point_num = setting_name.split('.')[1]
    #     if point_num not in known_points:
    #         known_points[point_num] = {}
    #
    #     if figure['figure'] is not None:
    #       box_center = (
    #         ((figure['figure']['points'][1][0] - figure['figure']['points'][0][0]) / 2
    #          + figure['figure']['points'][0][0]) * width,
    #         ((figure['figure']['points'][1][1] - figure['figure']['points'][0][1]) / 2
    #          + figure['figure']['points'][0][1]) * height
    #       )
    #
    #       known_points[point_num]['pixel'] = box_center
    #
    #   if re.search(r"externalprocessor.point\d*.latitude", setting_name):
    #     point_num = setting_name.split('.')[1]
    #     if point_num not in known_points:
    #         known_points[point_num] = {}
    #
    #     if 'lat_lon' not in known_points[point_num]:
    #       known_points[point_num]['lat_lon'] = (0,0)
    #
    #     known_points[point_num]['lat_lon'] = (setting_value, known_points[point_num]['lat_lon'][1])
    #
    #   if re.search(r"externalprocessor.point\d*.longitude", setting_name):
    #     point_num = setting_name.split('.')[1]
    #     if point_num not in known_points:
    #       known_points[point_num] = {}
    #
    #     if 'lat_lon' not in known_points[point_num]:
    #       known_points[point_num]['lat_lon'] = (0, 0)
    #
    #     known_points[point_num]['lat_lon'] = (known_points[point_num]['lat_lon'][0], setting_value)
    #
    #   if known_points != known_points_cache and len(known_points) >= 4:
    #     H = compute_homography(known_points)
    #
    #     known_points_cache = known_points
    #
    #   if H is not None:
    #
    #     for class_name, bboxes in input_object["BBoxes_xyxy"].items():
    #       object_index = 0
    #       coordinate_counter = 0
    #       bbox_pixel = [0, 0, 0, 0]
    #       for bbox_coordinate in bboxes:
    #         bbox_pixel[coordinate_counter] = bbox_coordinate
    #         coordinate_counter += 1
    #         if coordinate_counter == 4:
    #           bbox_center = (
    #             ((bbox_pixel[2] - bbox_pixel[0]) / 2 + bbox_pixel[0]),
    #             ((bbox_pixel[3] - bbox_pixel[1]) / 2 + bbox_pixel[1])
    #           )
    #           lat, lon = apply_homography(H, bbox_center)
    #
    #           input_object["ObjectsMetaData"][class_name]['AttributeKeys'][object_index].append("Latitude")
    #           input_object["ObjectsMetaData"][class_name]['AttributeKeys'][object_index].append("Longitude")
    #           input_object["ObjectsMetaData"][class_name]['AttributeValues'][object_index].append(str(lat))
    #           input_object["ObjectsMetaData"][class_name]['AttributeValues'][object_index].append(str(lon))
    #
    #           coordinate_counter = 0
    #           object_index += 1
    #
    #     formatted_unpacked_object = pformat(input_object)

    logger.info("Added event to output")

    # Write object back to string
    output_message = communication_utils.writeInferenceResults(input_object)

    # Send message back to runtime
    communication_utils.sendMessageOverConnection(connection, output_message)


def compute_homography(known_points):
  """Computes a homography transformation matrix using OpenCV."""

  pixel_points = []
  real_world_points = []

  for key, values in known_points.items():
    pixel_points.append(values['pixel'])
    real_world_points.append(values['lat_lon'])

  pixel_points = np.array(pixel_points, dtype=np.float32)
  real_world_points = np.array(real_world_points, dtype=np.float32)

  H, _ = cv2.findHomography(pixel_points, real_world_points, method=cv2.RANSAC)

  return H


def apply_homography(H, pixel_coord):
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
