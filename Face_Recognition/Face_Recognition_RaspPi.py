#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Julian Addison
@email: julianlukeaddison@gmail.com
@date: 28 Feb 2021
@project: Face Recognition Alert
"""

import argparse
import face_recognition
import cv2
import numpy as np
import os
from twilio.rest import Client
import time
from picamera.array import PiRGBArray  # Generates a 3D RGB array
from picamera import PiCamera  # Provides a Python interface for the RPi Camera Module


def send_sms(name_ls, face_count, verbose=False):
    account_sid = os.environ['TWILIO_ACCOUNT_SID']
    auth_token = os.environ['TWILIO_AUTH_TOKEN']
    client = Client(account_sid, auth_token)

    unique_name_ls = list(set(name_ls))
    if 'UNKNOWN' in unique_name_ls and face_count < 3:
        known_name_ls = unique_name_ls.remove('UNKNOWN')
        print('-------', ', '.join(known_name_ls), str(face_count-len(known_name_ls)))
        message_body = '{} and {} UNKNOWN people are here.'.format(', '.join(known_name_ls),
                                                                   str(face_count-len(known_name_ls)))
    else:
        message_body = '{} people are here.'.format(', '.join(unique_name_ls))

    message = client.messages.create(body=message_body,
                                     from_='+19164714674',
                                     to=os.environ['SG_PHONE_NUMBER']
                                     )
    if verbose:
        print(message.sid)
    return


def known_faces(foldername='images', verbose=False):
    known_face_encodings = []
    known_face_names = []

    image_list = os.listdir(foldername)
    for image in image_list:
        print(image.split('.')[0])

        img = face_recognition.load_image_file(os.path.join(foldername, image))
        face_encoding = face_recognition.face_encodings(img)[0]

        known_face_encodings.append(face_encoding)

        name = image.split('.')[0].upper()
        known_face_names.append(name)

        # unpack face coordinates
        if verbose:
            face_locations = face_recognition.face_locations(img)
            top, right, bottom, left = face_locations[0]

            frame = cv2.imread(os.path.join(foldername, image))
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            cv2.imwrite(os.path.join('output', name + '_detected.jpeg'), frame)
    return known_face_encodings, known_face_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Set camera option")
    parser.add_argument("-c", "--camera_type", type=str, default='webcam', help="Specify camera type")
    args = parser.parse_args()

    # Get known face encodings
    known_face_encodings, known_face_names = known_faces(foldername='images', verbose=False)

    # Initialize variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    send_sms_flag = 0

    if args.camera_type == 'webcam':
        # Get a reference to webcam #0 (the default one)
        video_capture = cv2.VideoCapture(0)

        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
            if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "UNKNOWN"

                    # # If a match was found in known_face_encodings, just use the first one.
                    # if True in matches:
                    #     first_match_index = matches.index(True)
                    #     name = known_face_names[first_match_index]

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    face_names.append(name)

            process_this_frame = not process_this_frame

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Add face counter at top right
            num_faces = len(face_locations)
            face_encodings_total = 'Total Face Count: {}'.format(str(num_faces))
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, face_encodings_total, (30, 30), font, 1.0, (0, 0, 0), 1)

            # Send SMS alert on who is here
            if send_sms_flag == 0 and num_faces > 1:
                send_sms(face_names, num_faces)
                send_sms_flag = 1

            # Display the resulting image
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if args.camera_type =='picamera':
        # Initialize the camera
        camera = PiCamera()

        # Set the camera resolution
        camera.resolution = (640, 480)

        # Set the number of frames per second
        camera.framerate = 32

        # Generates a 3D RGB array and stores it in rawCapture
        raw_capture = PiRGBArray(camera, size=(640, 480))

        # Wait a certain number of seconds to allow the camera time to warmup
        time.sleep(0.1)

        # Capture frames continuously from the camera
        for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):

            # Grab the raw NumPy array representing the image
            image = frame.array

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
            if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "UNKNOWN"

                    # # If a match was found in known_face_encodings, just use the first one.
                    # if True in matches:
                    #     first_match_index = matches.index(True)
                    #     name = known_face_names[first_match_index]

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    face_names.append(name)

            process_this_frame = not process_this_frame

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Add face counter at top right
            num_faces = len(face_locations)
            face_encodings_total = 'Total Face Count: {}'.format(str(num_faces))
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image, face_encodings_total, (30, 30), font, 1.0, (0, 0, 0), 1)

            # Display the frame using OpenCV
            cv2.imshow("Frame", image)

            # Wait for keyPress for 1 millisecond
            key = cv2.waitKey(1) & 0xFF

            # Clear the stream in preparation for the next frame
            raw_capture.truncate(0)

            # If the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
