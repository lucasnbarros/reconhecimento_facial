from cv2 import cv2  # biblioteca de computacao visual e aprendizado de maquina


# carrega o xml que contem um modelo pre-treinado de deteccao de rostos frontais
detector_rostos = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# carrega o xml que detecta olhos
detector_olhos = cv2.CascadeClassifier('haarcascade_eye.xml')

# carrega o xml que detecta olhos com oculos
# detector_olhos = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml') # para utilizar, basta descomentar essa linha

# reconhecimento em vídeo
# cap = cv2.VideoCapture('oculos1.mp4')

# reconhecimento pela webcam
cap = cv2.VideoCapture(0)

while 1:
    inf, quadro = cap.read()
    gray = cv2.cvtColor(quadro, cv2.COLOR_BGR2GRAY)
    rostos = detector_rostos.detectMultiScale(gray, 1.3, 5)

# procura os olhos dentro dos rostos
    for (x, y, w, h) in rostos:
        cv2.rectangle(quadro, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = quadro[y:y+h, x:x+w]

        olhos = detector_olhos.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in olhos:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    cv2.imshow('quadro', quadro)

    k = cv2.waitKey(10) & 0xff

    if k == 27:
        break

cap.release()

cv2.destroyAllWindows()
