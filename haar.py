import cv2

def load_default_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def load_alt_cascade(): 
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

if __name__ == "__main__":
    img = cv2.imread('face.png')
    img1 = cv2.imread('face.png')
    for (x, y, w, h) in load_alt_cascade().detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)):
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    for (x, y, w, h) in load_default_cascade().detectMultiScale(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)):
        cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 0, 0), 2)

    cv2.imshow('Detected Faces Alt', img)
    cv2.imshow('Detected Faces Default', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()