from PIL import Image
from os import path

import distutils.dir_util
import sys
import cv2


class Faces(object):
    def __init__(self, source_image):
        self.source_image = source_image
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.cv_image = cv2.imread(self.source_image)
        self.gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        self.faces = self.detect_multiscale()
        self.found = len(self.faces)

    def detect_multiscale(self):
        detected = self.face_cascade.detectMultiScale(
            self.gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )

        return detected

    def show_faces(self):
        # Draw a rectangle around the faces
        for (x, y, w, h) in self.faces:
            cv2.rectangle(self.cv_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Faces found', self.cv_image)
        cv2.waitKey(0)

    def extract_faces(self):
        if self.found:
            face_dir = path.join('thumbs', self.source_image.split('.')[0])
            distutils.dir_util.mkpath(face_dir)
            img = Image.open(self.source_image)
            for n, face in enumerate(self.faces):
                x, y, w, h = face
                thumb = img.crop((x, y, x+w, y+h))
                thumb.save(path.join(face_dir, f'face_{n}.jpg'))

            print(f'Found {self.found} faces!')
        else:
            print('No faces were detected!')


def main():
    # Get user supplied values
    if len(sys.argv) == 1:
        print('You must specify an image to process:\n')
        print('    python thumbler.py image.jpg\n')
    else:
        source_image = sys.argv[1]

        pic = Faces(source_image)
        pic.show_faces()
        pic.extract_faces()

if __name__ == '__main__':
    main()
