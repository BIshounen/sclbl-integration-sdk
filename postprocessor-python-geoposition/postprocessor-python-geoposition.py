import os
import sys
import socket
import signal
import logging
import logging.handlers
import configparser
from pprint import pformat
from affine_transform import get_pixel_to_coordinates

# Add the nxai-utilities python utilities
if getattr(sys, "frozen", False):
    script_location = os.path.dirname(sys.executable)
elif __file__:
    script_location = os.path.dirname(__file__)
sys.path.append(os.path.join(script_location, "../nxai-utilities/python-utilities"))
import communication_utils


CONFIG_FILE = os.path.join(script_location, "..", "etc", "plugin.settings.ini")

# Set up logging
LOG_FILE = os.path.join(script_location, "..", "etc", "plugin.settings.log")

# Initialize plugin and logging, script makes use of INFO and DEBUG levels
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - example - %(message)s",
    filename=LOG_FILE,
    filemode="w",
)

# The name of the postprocessor.
# This is used to match the definition of the postprocessor with routing.
Postprocessor_Name = "Python-GeoPosition-Postprocessor"

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

        # Read settings to get known points for affine transformation
        known_points = [
            {"pixel": (None, None), "lat_lon": (None, None)},
            {"pixel": (None, None), "lat_lon": (None, None)},
            {"pixel": (None, None), "lat_lon": (None, None)}
        ]

        for setting_name, setting_value in input_object[
            "ExternalProcessorSettings"
        ].items():
            lat1 = None
            lon1 = None
            lat2 = None
            lon2 = None
            lat3 = None
            lon3 = None
            if setting_name == "externalprocessor.point1":
                known_points[0]["pixel"] = (setting_value["figure"]["points"][0][0],
                                            setting_value["figure"]["points"][0][1])

            if setting_name == "externalprocessor.point2":
                known_points[1]["pixel"] = (setting_value["figure"]["points"][0][0],
                                            setting_value["figure"]["points"][0][1])

            if setting_name == "externalprocessor.point3":
                known_points[2]["pixel"] = (setting_value["figure"]["points"][0][0],
                                            setting_value["figure"]["points"][0][1])

            if setting_name == "externalprocessor.point1Latitude":
                lat1 = setting_value
            if setting_name == "externalprocessor.point1Longitude":
                lon1 = setting_value

            if setting_name == "externalprocessor.point2Latitude":
                lat2 = setting_value
            if setting_name == "externalprocessor.point2Longitude":
                lon2 = setting_value

            if setting_name == "externalprocessor.point3Latitude":
                lat3 = setting_value
            if setting_name == "externalprocessor.point3Longitude":
                lon3 = setting_value

            known_points[0]['lat_lon'] = (lat1, lon1)
            known_points[1]['lat_lon'] = (lat2, lon2)
            known_points[2]['lat_lon'] = (lat3, lon3)

        # Add lat and long to attributes
        for class_name, bboxes in input_object['BBoxes_xyxy'].items():
            for object_index in range(len(bboxes)):
                bbox = bboxes[object_index]
                central_pixel = (bbox[0] - bbox[2], bbox[1] - bbox[3])
                lat, lon = get_pixel_to_coordinates(known_points=known_points, pixel=central_pixel)
                lat = round(float(lat), 3)
                lon = round(float(lon), 3)

                input_object['ObjectsMetaData'][class_name]["AttributeKeys"][object_index].append("Latitude")
                input_object['ObjectsMetaData'][class_name]["AttributeKeys"][object_index].append(lat)
                input_object['ObjectsMetaData'][class_name]["AttributeKeys"][object_index].append("Longitude")
                input_object['ObjectsMetaData'][class_name]["AttributeKeys"][object_index].append(lon)

        formatted_unpacked_object = pformat(input_object)
        logging.info(f"Packing:\n\n{formatted_unpacked_object}\n\n")

        logger.info("Added attributes to all objects.")

        # Write object back to string
        output_message = communication_utils.writeInferenceResults(input_object)

        # Send message back to runtime
        communication_utils.sendMessageOverConnection(connection, output_message)


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
