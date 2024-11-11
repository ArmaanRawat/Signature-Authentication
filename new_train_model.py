import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

image_size = (128, 128)
batch_size = 32
epochs = 20
base_directory = r'C:\Users\yashm\Downloads\archive\sign_data\train'

def load_images_from_multiple_folders(base_directory):
    images, labels = [], []
    for signature_folder in os.listdir(base_directory):
        signature_path = os.path.join(base_directory, signature_folder)
        if os.path.isdir(signature_path):
            original_folder = os.path.join(signature_path, 'original')
            if os.path.exists(original_folder):
                original_images, original_labels = load_images_and_labels(original_folder, 0)
                images.extend(original_images)
                labels.extend(original_labels)
            forged_folder = os.path.join(signature_path, 'forged')
            if os.path.exists(forged_folder):
                forged_images, forged_labels = load_images_and_labels(forged_folder, 1)
                images.extend(forged_images)
                labels.extend(forged_labels)
    return np.array(images), np.array(labels)

def load_images_and_labels(directory, label):
    images, labels = [], []
    for filename in os.listdir(directory):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, image_size)
            img = img / 255.0
            images.append(img)
            labels.append(label)
    return images, labels

images, labels = load_images_from_multiple_folders(base_directory)
images = images.reshape(images.shape[0], image_size[0], image_size[1], 1)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
y_train, y_test = to_categorical(y_train, 2), to_categorical(y_test, 2)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
model.save(r"D:\signature_verification\signature_verification_model.keras")

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'train accuracy: {test_acc * 100:.2f}%')

def verify_signatures(model_path, test_directory):
    model = load_model(model_path)
    if not os.path.exists(test_directory):
        print(f"Error: The test directory {test_directory} does not exist.")
        return
    test_images, filenames = [], []
    for filename in os.listdir(test_directory):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(test_directory, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, image_size)
            img = img / 255.0
            img = img.reshape(1, image_size[0], image_size[1], 1)
            test_images.append(img)
            filenames.append(filename)
    for i, img in enumerate(test_images):
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        result = "Genuine" if predicted_class == 0 else "Forged"
        print(f"File: {filenames[i]}, Prediction: {result}, Confidence: {confidence:.2f}%")

test_directory = r"C:\Users\yashm\Downloads\signatures"
verify_signatures(r"D:\signature_verification\signature_verification_model.keras", test_directory)
