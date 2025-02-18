import os
import sys
import socket
import signal
import logging
import logging.handlers
import configparser
from pprint import pformat
import json

import numpy as np
from scipy.optimize import minimize

# Add the nxai-utilities python utilities
if getattr(sys, "frozen", False):
    script_location = os.path.dirname(sys.executable)
elif __file__:
    script_location = os.path.dirname(__file__)
sys.path.append(os.path.join(script_location, "../nxai-utilities/python-utilities"))
import communication_utils


CONFIG_FILE = os.path.join(script_location, "..", "etc", "plugin.geoposition.ini")

# Set up logging
LOG_FILE = os.path.join(script_location, "..", "etc", "plugin.geoposition.log")

# Initialize plugin and logging, script makes use of INFO and DEBUG levels
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - example - %(message)s",
    filename=LOG_FILE,
    filemode="w",
)

# The name of the postprocessor.
# This is used to match the definition of the postprocessor with routing.
Postprocessor_Name = "Python-Geoposition-Postprocessor"

# The socket this postprocessor will listen on.
# This is always given as the first argument when the process is started
# But it can be manually defined as well, as long as it is the same as the socket path in the runtime settings
Postprocessor_Socket_Path = "/tmp/python-geoposition-postprocessor.sock"

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
    logger.debug("Creating socket at " + Postprocessor_Socket_Path)
    server = communication_utils.startUnixSocketServer(Postprocessor_Socket_Path)

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

        # Use pformat to format the deep object
        formatted_unpacked_object = pformat(input_object)
        logging.info(f"Unpacked:\n\n{formatted_unpacked_object}\n\n")

        lat_lon = {}

        known_points = {
            "pixels": [(None, None), (None, None), (None, None)],
            "lat_lon": [(None, None), (None, None), (None, None)]
        }

        # coefficient for translating to real coordinates because of the Nx three digits mantissa
        mantissa_coefficient = 1000

        device_id = input_object.get("DeviceID", "")

        for setting_name, setting_value in input_object["ExternalProcessorSettings"].items():

            # Get pixel coordinates
            if setting_name in ("externalprocessor.point1", "externalprocessor.point2", "externalprocessor.point3"):
                figure = json.loads(setting_value)
                box_center = (
                    (figure['figure']['points'][1][0] - figure['figure']['points'][0][0])/2
                    + figure['figure']['points'][0][0],
                    (figure['figure']['points'][1][1] - figure['figure']['points'][0][1]) / 2
                    + figure['figure']['points'][0][1]
                )

                if setting_name == "externalprocessor.point1":
                    known_points['pixels'][0] = box_center
                elif setting_name == "externalprocessor.point2":
                    known_points['pixels'][1] = box_center
                else:
                    known_points['pixels'][2] = box_center

            if setting_name == "externalprocessor.point1Latitude":
                lat_lon['lat1'] = float(setting_value) * mantissa_coefficient
            if setting_name == "externalprocessor.point1Longitude":
                lat_lon['lon1'] = float(setting_value) * mantissa_coefficient
            if setting_name == "externalprocessor.point2Latitude":
                lat_lon['lat2'] = float(setting_value) * mantissa_coefficient
            if setting_name == "externalprocessor.point2Longitude":
                lat_lon['lon2'] = float(setting_value) * mantissa_coefficient
            if setting_name == "externalprocessor.point3Latitude":
                lat_lon['lat3'] = float(setting_value) * mantissa_coefficient
            if setting_name == "externalprocessor.point3Longitude":
                lat_lon['lon3'] = float(setting_value) * mantissa_coefficient

        known_points['lat_lon'][0] = (lat_lon['lat1'], lat_lon['lon1'])
        known_points['lat_lon'][1] = (lat_lon['lat2'], lat_lon['lon2'])
        known_points['lat_lon'][2] = (lat_lon['lat3'], lat_lon['lon3'])

        logger.info(f"Got known point coordinates from settings: {known_points}")

        # get image parameters
        width = input_object['Width']
        height = input_object['Height']

        for class_name, bboxes in input_object["BBoxes_xyxy"].items():
            object_index = 0
            coordinate_counter = 0
            bbox_pixel = [0, 0, 0, 0]
            for bbox_coordinate in bboxes:
                bbox_pixel[coordinate_counter] = bbox_coordinate
                coordinate_counter += 1
                if coordinate_counter == 4:

                    matrix = compute_transformation_matrix(pixel_points=known_points['pixels'],
                                                           real_world_points=known_points['lat_lon'])

                    bbox_center = (
                        (bbox_pixel[2] - bbox_pixel[0])/2 + bbox_pixel[0],
                        (bbox_pixel[3] - bbox_pixel[1])/2 + bbox_pixel[1]
                    )
                    lat, lon = apply_transformation(T=matrix, pixel_coord=bbox_center)

                    input_object[class_name]['AttributeKeys'][object_index].append("Latitude")
                    input_object[class_name]['AttributeKeys'][object_index].append("Longitude")
                    input_object[class_name]['AttributeValues'][object_index].append(lat)
                    input_object[class_name]['AttributeValues'][object_index].append(lon)

                    coordinate_counter = 0
                    object_index += 1


        # Read the settings passed through from the AI Manager and add them as attributes
        for _, class_data in input_object["ObjectsMetaData"].items():
            for object_index in range(len(class_data["AttributeKeys"])):
                for setting_name, setting_value in input_object[
                    "ExternalProcessorSettings"
                ].items():
                    if setting_name == "externalprocessor.attributeName":
                        class_data["AttributeKeys"][object_index].append(setting_value)
                    if setting_name == "externalprocessor.attributeValue":
                        class_data["AttributeValues"][object_index].append(
                            setting_value
                        )

        formatted_unpacked_object = pformat(input_object)
        logging.info(f"Packing:\n\n{formatted_unpacked_object}\n\n")

        logger.info("Added attributes to all objects.")

        # Write object back to string
        output_message = communication_utils.writeInferenceResults(input_object)

        # Send message back to runtime
        communication_utils.sendMessageOverConnection(connection, output_message)


# def affine_transform(params, pixel_coordinates, lat_lon_coordinates):
#   a, b, c, d, e, f = params  # Affine parameters
#   pixel_x, pixel_y = pixel_coordinates.T  # Transpose to get x and y
#   # Calculate lat/lon from the affine parameters
#   lat = a * pixel_x + b * pixel_y + c
#   lon = d * pixel_x + e * pixel_y + f
#   # Return the residual (difference) between calculated and actual lat/lon
#   return np.concatenate([lat - lat_lon_coordinates[:, 0], lon - lat_lon_coordinates[:, 1]])
#
# def transform_point(pixel, params):
#   a, b, c, d, e, f = params
#   pixel_x, pixel_y = pixel
#   lat = a * pixel_x + b * pixel_y + c
#   lon = d * pixel_x + e * pixel_y + f
#   return lat, lon


# def get_pixel_to_coordinates(known_points, pixel) -> tuple[float, float]:
#   pixel_coordinates = np.array([p["pixel"] for p in known_points])
#   lat_lon_coordinates = np.array([p["lat_lon"] for p in known_points])
#
#   initial_params = [0, 0, 0, 0, 0, 0]  # Initial guess
#   result = minimize(
#     lambda params: np.sum(affine_transform(params, pixel_coordinates, lat_lon_coordinates) ** 2),
#     initial_params,
#   )
#   # Extract optimized parameters
#   optimized_params = result.x
#
#   return transform_point(pixel, optimized_params)


def compute_transformation_matrix(pixel_points, real_world_points):
    """
    Computes the affine transformation matrix from pixel coordinates to real-world coordinates.
    pixel_points: List of 3 (x, y) pixel coordinates.
    real_world_points: List of 3 (X, Y) real-world coordinates.
    Returns: 2x3 affine transformation matrix.
    """
    A = []
    B = []

    for (px, py), (wx, wy) in zip(pixel_points, real_world_points):
        A.append([px, py, 1, 0, 0, 0])
        A.append([0, 0, 0, px, py, 1])
        B.append(wx)
        B.append(wy)

    A = np.array(A)
    B = np.array(B)

    # Solve for the transformation matrix
    T = np.linalg.lstsq(A, B, rcond=None)[0]
    return T.reshape(2, 3)


def apply_transformation(T, pixel_coord):
    """
    Applies the affine transformation matrix to a given pixel coordinate.
    T: 2x3 affine transformation matrix.
    pixel_coord: (x, y) pixel coordinate.
    Returns: (X, Y) real-world coordinate.
    """
    px, py = pixel_coord
    X = T[0, 0] * px + T[0, 1] * py + T[0, 2]
    Y = T[1, 0] * px + T[1, 1] * py + T[1, 2]
    return (X, Y)


if __name__ == "__main__":
    ## initialize the logger
    logger = logging.getLogger(__name__)

    logger.info("Location: " + str(script_location))

    ## read configuration file if it's available
    config()

    logger.info("Initializing example plugin")
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
