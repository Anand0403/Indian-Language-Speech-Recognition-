# Mount Google Drive to access files stored in your drive
from google.colab import drive
drive.mount('/content/drive')


# Install required libraries
%pip install librosa resampy matplotlib seaborn keras scikit-learn
import os
import librosa
import librosa.display
import resampy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

# Function to extract mel spectrogram from audio file
def extract_mel_spectrogram(audio_path, max_pad_len=174):
    audio, sr = librosa.load(audio_path, res_type='kaiser_fast')
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_resized = librosa.util.fix_length(mel_spec_db, size=max_pad_len, axis=1)
    return mel_spec_resized

# Function to load and process audio data for training or testing
def load_data(input_folder):
    data = []
    labels = []
    class_subdirs = [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]
    
    # Process each class folder
    for i, class_dir in enumerate(class_subdirs):
        class_folder = os.path.join(input_folder, class_dir)
        for file in os.listdir(class_folder):
            if file.endswith('.wav'):
                file_path = os.path.join(class_folder, file)
                try:
                    audio, _ = librosa.load(file_path, res_type='kaiser_fast')
                    if len(audio) > 0:
                        mel_spec = extract_mel_spectrogram(file_path)
                        data.append(mel_spec)
                        labels.append(class_dir)
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")

    return np.array(data), np.array(labels)

# Function to process a single audio file
def load_single_file(file_path):
    mel_spec = extract_mel_spectrogram(file_path)
    return np.expand_dims(mel_spec, axis=0)  # Add batch dimension

# Build CNN model for classification
def build_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(*input_shape, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(5, activation='softmax'))  # 5 classes for classification
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Set input folders for training and testing
    train_folder = "/content/drive/MyDrive/project3/train"
    test_path = "/content/drive/MyDrive/project3/test" 

    # Load training data
    X_train, y_train = load_data(train_folder)

    # Encode labels as categorical
    label_encoder = LabelEncoder()
    y_train_encoded = to_categorical(label_encoder.fit_transform(y_train))

    # Reshape input data for CNN
    input_shape = X_train[0].shape  # Shape of the mel spectrogram
    X_train = X_train.reshape(X_train.shape[0], *input_shape, 1)

    # Build and train the model
    model = build_cnn_model(input_shape)
    model.fit(X_train, y_train_encoded, epochs=10, batch_size=32)

    # Test the model on a single file or the entire test set
    if os.path.isfile(test_path):
        X_test = load_single_file(test_path)
        X_test = X_test.reshape(X_test.shape[0], *input_shape, 1)
        predictions = model.predict(X_test)
        y_pred = np.argmax(predictions, axis=1)
        
        class_accuracies = {label: 0 for label in label_encoder.classes_}
        predicted_class = label_encoder.classes_[y_pred[0]]
        class_accuracies[predicted_class] = 100.0

        # Print the predicted class and its accuracy
        print(f"Predicted class index: {y_pred[0]}")
        print(f"Predicted class label: {predicted_class}")
        print("Class Accuracies:")
        for language, accuracy in class_accuracies.items():
            print(f"{language}: {accuracy}%")
    else:
        X_test, y_test = load_data(test_path)
        y_test_encoded = to_categorical(label_encoder.transform(y_test))
        X_test = X_test.reshape(X_test.shape[0], *input_shape, 1)

        # Evaluate the model on the test set
        test_loss, test_acc = model.evaluate(X_test, y_test_encoded)
        print(f"Test Accuracy: {test_acc * 100:.2f}%")

        # Calculate and print per-class accuracies
        predictions = model.predict(X_test)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test_encoded, axis=1)

        class_accuracies = {}
        for class_label in range(len(label_encoder.classes_)):
            class_indices = np.where(y_true == class_label)
            class_accuracy = np.sum(y_pred[class_indices] == y_true[class_indices]) / len(class_indices[0])
            class_accuracies[label_encoder.classes_[class_label]] = class_accuracy * 100  # Convert to percentage

        print("Class Accuracies:")
        for language, accuracy in class_accuracies.items():
            print(f"{language}: {accuracy:.2f}%")



# Plot the confusion matrix to evaluate model performance
%pip install seaborn matplotlib
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

