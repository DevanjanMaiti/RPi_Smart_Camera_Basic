1. Follow the steps in Raspberry_Pi_Guide.md from Step 1a till Step 1c.
2. Run Step 1d using Option 1; model .zip can be downloaded directly from this repo in case you face any challenges with Google storage API.
3. Replace the "TFLite_detection_webcam.py" file in "tflite1" directory with the version available in this repository.
4. Download and install rclone to sync files to Google Drive using the following commands:
        sudo snap install snapd
        sudo reboot
        sudo snap install rclone
5. Run rclone setup using the command "rclone config" and configure your client to connect to Google Drive API by creating a new remote.
     Note that a new remote expires after 7 days of use.
6.  Run step 1e as is; or else download the cam_script.sh file and run the shell script from your /home/pi area itself.
