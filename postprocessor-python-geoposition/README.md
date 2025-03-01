# Post-Processor for AI Manager

This post-processor converts bounding boxes into real-world coordinates for AI Manager.

## Usage Instructions

- Clone the repository
   ```sh
   git clone https://github.com/scailable/sclbl-integration-sdk.git --recurse-submodules
   ```
   If you have downloaded the sdk previously, you can also update to the latest version of the integration SDK while in the directory of the downloaded git repository.
   ```sh
   git pull --recurse-submodules
   ```
- Install the needed dependencies:
   ```sh
   sudo apt install cmake
   sudo apt install g++
   sudo apt install python3-pip
   sudo apt install python3.12-venv
   ```
  **This plugin also uses opencv python library, so additional dependencies might be needed to make the plugin. Check build logs for any unresolved dependencies.**


- Change into the directory created for the project if you're not already there.
   ```shell
   cd sclbl-integration-sdk/
   ```
- Prepare the build directory in the project directory, and switch to the build directory.
   ```shell
   mkdir -p build
   cd build
   ```
- Create and activate a virtual environment:
   ```sh
   python3 -m venv integrationsdk
   source integrationsdk/bin/activate  # On Windows use 'venv\Scripts\activate'
   ```
- Build the postprocessor, while in the created build directory. This may take a while, depending on the speed of your system.
   ```sh
   сmake ..
   make
   ```
- Copy the necessary files to the appropriate location. A convenience directory within the Edge AI Manager installation is created for this purpose at `/opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin/nxai_manager/postprocessors`, but you can specify any directory in the external-postprocessors.json.

- **Create an INI file** based on the template in the `../etc/` directory relative to the post-processor path.
   - This file must include the RabbitMQ server address where messages will be sent
   ```sh
   [common]
   debug_level=INFO
   
   [mq]
   address=localhost
   port=5672
   login=guest
   password=guest
   ```
- Configure **external post-processors**:
   - Add the settings found in `external_postprocessors.json` to the corresponding section.
- Restart the server.
- Start the **AI Manager** and configure the pipeline with the post-processor according to its manual: [AI Manager Guide](https://nx.docs.scailable.net/nx-ai-manager/get-started-with-the-nx-ai-manager-plugin).
- In the post-processor settings, specify four reference points that map camera pixels to real-world **latitude/longitude** coordinates. Due to limitations of Nx desktop client settings, values must be entered multiplied by 1000, e.g. 33.878746 must be entered as 33878.756, due to 3 digits precision limitation:
![alt text](https://github.com/BIshounen/sclbl-integration-sdk/blob/main/postprocessor-python-geoposition/readme_images/settings.png?raw=true)

# Troubleshooting

### Plugin doesn't start

The plugin might be built without resolving opencv dependencies, try to build from scratch and check logs. If dependencies are not met, use `sudo apt install` to install necessary dependencies.
