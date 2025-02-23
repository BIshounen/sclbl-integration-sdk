import os
import sys
import socket
import signal
import logging
import logging.handlers
import configparser
from pprint import pformat
import json
import uuid
import pika

# import numpy as np
# import cv2
# import queue
# import threading

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

# Address of the ZMQ server to send data to
mq_server = "tcp://127.0.0.1:5555"
pika_login = "guest"
pika_password = "guest"
pika_port = 5672

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

        global mq_server
        mq_server = configuration.get("mq", "address", fallback="localhost")
        global pika_login
        pika_login = configuration.get("mq", "login", fallback="guest")
        global pika_password
        pika_password = configuration.get("mq", "password", fallback="guest")
        global pika_port
        pika_port = configuration.get("mq", "port", fallback="5672")

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

    # start MQ
    credentials = pika.PlainCredentials(pika_login, pika_password)
    parameters = pika.URLParameters(mq_server)
    # parameters = pika.ConnectionParameters(mq_server, pika_port, '/', credentials)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    channel.queue_declare(queue="AIManager")

    # data_queue = queue.Queue()
    H = None

    known_points_cache = {
        "pixels": [(None, None), (None, None), (None, None), (None, None)],
        "lat_lon": [(None, None), (None, None), (None, None), (None, None)]
    }

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
            "pixels": [(None, None), (None, None), (None, None), (None, None)],
            "lat_lon": [(None, None), (None, None), (None, None), (None, None)]
        }

        # coefficient for translating to real coordinates because of the Nx three digits mantissa
        mantissa_coefficient = 1000

        device_id = input_object.get("DeviceID", "")
        timestamp = input_object.get("Timestamp", "")

        message = {"device_id": device_id, "timestamp": timestamp}

        for setting_name, setting_value in input_object["ExternalProcessorSettings"].items():

            # get image parameters
            width = input_object['Width']
            height = input_object['Height']

            # Get pixel coordinates
            if setting_name in ("externalprocessor.point1",
                                "externalprocessor.point2",
                                "externalprocessor.point3",
                                "externalprocessor.point4"
                                ):
                figure = json.loads(setting_value)
                box_center = (
                    ((figure['figure']['points'][1][0] - figure['figure']['points'][0][0])/2
                    + figure['figure']['points'][0][0]) * width,
                    ((figure['figure']['points'][1][1] - figure['figure']['points'][0][1]) / 2
                    + figure['figure']['points'][0][1]) * height
                )

                if setting_name == "externalprocessor.point1":
                    known_points['pixels'][0] = box_center
                elif setting_name == "externalprocessor.point2":
                    known_points['pixels'][1] = box_center
                elif setting_name == "externalprocessor.point3":
                    known_points['pixels'][2] = box_center
                else:
                    known_points['pixels'][3] = box_center

            if setting_name == "externalprocessor.point1Latitude":
                lat_lon['lat1'] = float(setting_value) / mantissa_coefficient
            if setting_name == "externalprocessor.point1Longitude":
                lat_lon['lon1'] = float(setting_value) / mantissa_coefficient
            if setting_name == "externalprocessor.point2Latitude":
                lat_lon['lat2'] = float(setting_value) / mantissa_coefficient
            if setting_name == "externalprocessor.point2Longitude":
                lat_lon['lon2'] = float(setting_value) / mantissa_coefficient
            if setting_name == "externalprocessor.point3Latitude":
                lat_lon['lat3'] = float(setting_value) / mantissa_coefficient
            if setting_name == "externalprocessor.point3Longitude":
                lat_lon['lon3'] = float(setting_value) / mantissa_coefficient
            if setting_name == "externalprocessor.point4Latitude":
                lat_lon['lat4'] = float(setting_value) / mantissa_coefficient
            if setting_name == "externalprocessor.point4Longitude":
                lat_lon['lon4'] = float(setting_value) / mantissa_coefficient

        known_points['lat_lon'][0] = (lat_lon['lat1'], lat_lon['lon1'])
        known_points['lat_lon'][1] = (lat_lon['lat2'], lat_lon['lon2'])
        known_points['lat_lon'][2] = (lat_lon['lat3'], lat_lon['lon3'])
        known_points['lat_lon'][3] = (lat_lon['lat4'], lat_lon['lon4'])

        logger.info(f"Got known point coordinates from settings: {known_points}")

        if known_points != known_points_cache:
            H = None
            # compute_thread = threading.Thread(
            #     target=lambda: compute_homography(data_queue, known_points['pixels'], known_points['lat_lon']))
            #
            # compute_thread.start()

        # if H is None and data_queue.not_empty:
        #     H = data_queue.get()
        #     logging.info('got H')

        # if H is not None:
        #
        #     message['objects'] = []
        #
        #     for class_name, bboxes in input_object["BBoxes_xyxy"].items():
        #         object_index = 0
        #         coordinate_counter = 0
        #         bbox_pixel = [0, 0, 0, 0]
        #         for bbox_coordinate in bboxes:
        #             bbox_pixel[coordinate_counter] = bbox_coordinate
        #             coordinate_counter += 1
        #             if coordinate_counter == 4:
        #
        #                 bbox_center = (
        #                     ((bbox_pixel[2] - bbox_pixel[0])/2 + bbox_pixel[0]),
        #                     ((bbox_pixel[3] - bbox_pixel[1])/2 + bbox_pixel[1])
        #                 )
        #                 lat, lon = apply_homography(H, bbox_center)
        #
        #                 input_object["ObjectsMetaData"][class_name]['AttributeKeys'][object_index].append("Latitude")
        #                 input_object["ObjectsMetaData"][class_name]['AttributeKeys'][object_index].append("Longitude")
        #                 input_object["ObjectsMetaData"][class_name]['AttributeValues'][object_index].append(str(lat))
        #                 input_object["ObjectsMetaData"][class_name]['AttributeValues'][object_index].append(str(lon))
        #
        #                 object_id = input_object["ObjectsMetaData"][class_name]['ObjectIDs'][object_index]
        #                 object_data = {
        #                     "type": class_name,
        #                     "object_id": str(uuid.UUID(bytes=object_id)),
        #                     "latitude": lat,
        #                     "longitude": lon
        #                 }
        #
        #                 message['objects'].append(object_data)
        #
        #                 coordinate_counter = 0
        #                 object_index += 1
        #
        #     channel.basic_publish(exchange='',
        #                           routing_key='AIManager',
        #                           body=json.dumps(message))
        #
        #     formatted_unpacked_object = pformat(input_object)
        #     logging.info(f"Packing:\n\n{formatted_unpacked_object}\n\n")
        #
        #     logger.info("Added attributes to all objects.")

        # Write object back to string
        output_message = communication_utils.writeInferenceResults(input_object)

        # Send message back to runtime
        communication_utils.sendMessageOverConnection(connection, output_message)


# def compute_homography(data_queue, pixel_points, real_world_points):
#     """Computes a homography transformation matrix using OpenCV."""
#     pixel_points = np.array(pixel_points, dtype=np.float32)
#     real_world_points = np.array(real_world_points, dtype=np.float32)
#
#     # H, _ = cv2.findHomography(pixel_points, real_world_points, method=cv2.RANSAC)
#     H = None
#     data_queue.put(H)
#     data_queue.task_done()


# def apply_homography(H, pixel_coord):
#     """Applies a homography transformation to a pixel coordinate."""
#     px, py = pixel_coord
#     transformed = np.dot(H, np.array([px, py, 1]))
#     X, Y = transformed[:2] / transformed[2]  # Normalize by Z
#     return (X, Y)


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
