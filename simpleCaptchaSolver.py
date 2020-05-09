import os.path
import os
import glob
import imutils
import cv2
import pickle
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from keras.models import load_model


### kiszedi a megadott CAPTCHA-kból a karaktereket es lementi azokat ###
def GenerateCharacterrs():
    print("Karakterek kigyujtese a Captcha-bol")
    captcha_image_files = glob.glob(os.path.join("captcha_images", "*"))
    counts = {}
    for (i, captcha_image_file) in enumerate(captcha_image_files):  # ciklus ami vegig megy az osszes training image-n
        filename = os.path.basename(captcha_image_file)
        captcha_correct_text = os.path.splitext(filename)[0]  # mekeresei az adott file nevet
        image = cv2.imread(captcha_image_file)  # beolvassa a képet, majd atalakitja grayscale-re
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] #threshold, azert hogy megtalalhassa a karaktereket mint contour
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # contour-ok megkeresése
        contours = contours[1] if imutils.is_cv3() else contours[0]
        letter_image_regions = []  # karakterek helyei, ehhez addodnak hozza a karakterek regioi
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)  # karakterek koruli bounding box koordinatak
            letter_image_regions.append((x, y, w, h))
        letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])  # karakter regiok rendezese
        for letter_bounding_box, letter_text in zip(letter_image_regions,
                                                    captcha_correct_text):  # ciklus ami vegigmegy a boxokon
            x, y, w, h = letter_bounding_box
            letter_image = gray[y - 2:y + h + 2,
                           x - 2:x + w + 2]  # egy uj kep lesz a karakterekbol a koordinatak alapjan, majd ezeket lementi(minden karakter kap egy folder-t es azon belül szamozva kerulnek az uj kepek a karakterekrol)
            save_path = os.path.join("generated_contours", letter_text)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            count = counts.get(letter_text, 1)  # szamozas
            p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
            cv2.imwrite(p, letter_image)
            counts[letter_text] = count + 1
    print("Karakterek kigyujtese vegetert")


### a kiszedett karaktereket tolti be es dolgozza fel ###
def LoadData():
    print("Kigyujtott adatok feldolgozasa")
    for image_file in paths.list_images("generated_contours"):  # Betolti a kiszedett karakterkepeket
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Atvaltja gray-re
        image = cv2.resize(image, (20, 20))  # Ha nem megfelelo a meret akkor a fenti segedfuggvennyel korrigalja
        image = np.expand_dims(image,
                               axis=2)  # Megnoveli a dim-ek szamat, ez csak azert kellett hogy passzoljon a network-be
        label = image_file.split(os.path.sep)[-2]  # kiszedi a file nevet, hogy label-t csinalhasson belole
        data.append(image)  # hozzaadja magat a kepeket
        labels.append(label)  # hozzaadja a labeleket

    print("Kigyujtott adatok feldolgozva")


### One-Hot labeleket keszit a korabban kiszedet labelekbol hogy konnyen hasznalhato legyen a network-be
def MakeOneHot(Y_train, Y_test):
    print("One-hot labelek elkeszitese")  # bemenetnek megkapja a Y_train-t és Y_test-t(test es train labelek)
    lb = LabelBinarizer().fit(Y_train)
    Y_train = lb.transform(Y_train)  # atalakitas
    Y_test = lb.transform(Y_test)
    with open("one_hot_labels.dat", "wb") as f:
        pickle.dump(lb, f)
    print("One-hot labelek elkeszitve")
    return Y_train, Y_test  # visszaadja az elkeszult one-hot labeleket


### Maga a Neural Network felepitese es trainelese ###
def BuildAndTrainNetwork():
    print("NN felepitese es trainelese")
    model.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1),
                     activation="relu"))  # Conc layer relu activation funtionnel #1
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # maxpooling layer #1
    model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))  # Conc layer relu activation funtionnel #2
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # maxpooling layer #2
    model.add(Flatten())  # egy flatten layer (input->output)
    model.add(Dense(500, activation="relu"))
    model.add(Dense(32, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", "mse"])
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=50, epochs=2,
              verbose=1)  # traineles, 2 epochs van, mivel mar a 2. epochnal eler egy adott accuracy-t, utana pedig nem valtozott jelentosen
    model.save("captcha_model.hdf5")  # lementi egy file-ba a modelt, kesobb konnyebb legyen hasznalni
    print("Neural network trainelese kesz")


### Egy teszt ami random kivalaszt 10 kepet es azokon teszteli a network-t ###
def RunTest(testsize):
    print("Teszt kezdete")
    counter=0
    f = open("one_hot_labels.dat", "rb")
    lb = pickle.load(f)
    network = load_model("captcha_model.hdf5")
    captcha_image_files = list(paths.list_images("captcha_images"))
    captcha_image_files = np.random.choice(captcha_image_files, size=(testsize,), replace=False)  # random kepek kivalasztasa

    for image_file in captcha_image_files:  # hasonloan a fenti GenerateCharacters function-hoz kigyujti a karaktereket a kepekbol
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[1] if imutils.is_cv2() else contours[0]
        letter_image_regions = []
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            letter_image_regions.append((x, y, w, h))
        letter_image_regions = sorted(letter_image_regions,
                                      key=lambda x: x[0])  # idaig csak a karakterek/contour-k kigyujtese
        predictions = []  # a predictionoket tarolja majd ebbe
        for letter_bounding_box in letter_image_regions:
            x, y, w, h = letter_bounding_box
            try:
                letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]
                letter_image = cv2.resize(letter_image, (20, 20))  # meret korigalasa
                letter_image = np.expand_dims(letter_image, axis=2)  # dimenziok kiigazitasa
                letter_image = np.expand_dims(letter_image, axis=0)
                prediction = network.predict(letter_image)      #predikcio
                letter = lb.inverse_transform(prediction)[0]
                predictions.append(letter)  # karakter prediction
            except cv2.error as e:
                print("Hiba a karakter keppel")
        captcha_text = "".join(predictions)
        print("Valaszott kep: {}".format(image_file))
        print("Predikcio: {}".format(captcha_text))

        base=os.path.basename(image_file)
        print("Correct: {}".format(captcha_text == os.path.splitext(base)[0]))
        if captcha_text==os.path.splitext(base)[0]:
            counter+=1
    print("Teszt vege")
    print("Eredmenyek: "+str(counter)+"/"+str(testsize)+" -> "+str(counter/testsize*100)+"%")


#########ures keras model, kesobb ez lesz bovitve:
#model = Sequential()

#########karakterek kigyujtese:
#GenerateCharacterrs()

#########karakterek betoltese/feldolgozasa:
data = []
labels = []
LoadData()
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
#########one-hot labelek elkeszitese:

(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)
(Y_train, Y_test) = MakeOneHot(Y_train, Y_test)

#########Network train:
#BuildAndTrainNetwork()

#########Teszt:
RunTest(100)
