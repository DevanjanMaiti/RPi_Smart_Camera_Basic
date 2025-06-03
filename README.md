This project uses a cheap RPi3 and Pi Camera v1 to create a low-cost replacement for commercially available cameras that offer only motion detection (faulty!) and expensive cloud storage for months of recording.

Instead, this project uses a light-weight TFLite object recognition model that detects multiple objects such as persons or cars and triggers recording of images or videos only when objects of interest are detected. These images/videos are then uploaded to a connected Google Drive account from where user can monitor relevant activities.

This project was created since commercially available cameras today don't have this basic brain and keep sending notifications for any random movement in its field of view. This creates unnecessary triggers to the user and makes finding real incidents a nightmare.

Please follow the steps below to set up your own simple Rpi Smart Camera:

1. Follow the steps in Raspberry_Pi_Guide.md from Step 1a till Step 1c.
2. Install the Image and imagehash library for image comparison used in the code by running: sudo pip3 install imagehash
3. Run Step 1d using Option 1; model .zip can be downloaded directly from this repo in case you face any challenges with Google storage API.
4. Replace the "TFLite_detection_webcam.py" file in "tflite1" directory with the version available in this repository.
5. Download and install rclone to sync files to Google Drive using the following commands:
        sudo snap install snapd
        sudo reboot
        sudo snap install rclone
6. Run rclone setup using the command "rclone config" and configure your client to connect to Google Drive API by creating a new remote.
     Note that a new remote expires after 7 days of use.
7.  Run step 1e as is; or else download the cam_script.sh file and run the shell script from your /home/pi area itself.
