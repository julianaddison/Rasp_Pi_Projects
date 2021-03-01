# Face Detection on RaspPi
Got tired of trying to figure out who was at the door when it rang. Decided to use a RaspPi to recognize faces
and send me alerts. This was more about deployment and playing around with the RaspPi setup. Essentially the
steps are:

1.  Detect number of faces in frame
2.  Check each face against a list of known faces
3.  Send an alert of the known and unknown characters

See this link on how it should all works out. _to be added_

It includes some basic performance tweaks to make things run faster on the RaspPi:

1. Process each video frame at 1/4 resolution (though still display it at full resolution)
2. Only detect faces in every other frame of video.

### Requirements
## Hardware
- Raspberry Pi 4 Model B
- Raspberry Pi 8MP Camera

## Additional Account/s & Setup
- Setup a Twilio account for experimentation. 
- Get your Twilio account sid and authorisation token and set them as environment  variables TWILIO_ACCOUNT_SID
 and TWILIO_AUTH_TOKEN. 
- Set your local phone number as an environment variable, SG_PHONE_NUMBER

## Packages
- OpenCV2
- numpy
- twilio 
- picamera
- Face Recognition _Follow https://pypi.org/project/face-recognition/ for pip installation instructions_

### Run
Run the following in bash. Use the camera type argument to set the camera in use i.e. "webcam" or "picamera". 

Usage:
```bash
python Face_Recognition_Webcam.py --camera_type "picamera"
```

### Data
The directory should look like this:
````bash
$ROOT/
|── Face_Recognition_RaspPi.py
├── images
|   |── xxxx.jpeg
│   └── jack.jpeg 
├── ouptut
│   ├── XXXX_detected.jpeg
│   └── JACK_detected.jpeg
````

### References
https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py
https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_on_raspberry_pi.py
http://blog.dlib.net/2017/02/high-quality-face-recognition-with-deep.html
https://www.twilio.com/docs/sms/quickstart/python