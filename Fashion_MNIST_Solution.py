# %% LIBRARY IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import confusion_matrix,classification_report
from sklearn.utils import class_weight


# %% LOAD DATASET

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# %% DATA VISUALIZATION

class_labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

fig, axes = plt.subplots(1, 6, figsize=(30, 24))

for i in range(6):
    axes[i].imshow(x_train[i], cmap="gray")

    label = class_labels[y_train[i]]
    axes[i].set_title(label)

    axes[i].axis("off")

plt.show()

# %% NORMALIZATION
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# %% ADD CHANNEL DIMENSION (RESHAPE)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# %% ONE-HOT ENCODING

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# %% DATA AUGMENTATION

datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode="nearest"
)

datagen.fit(x_train)

model = Sequential()

model.add(Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=x_train.shape[1:]))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.35))

model.add(Flatten())

model.add(Dense(256, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(10, activation="softmax"))
# %% CALCULATE CLASS WEIGHTS




y_train_labels = np.argmax(y_train, axis=1)


class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_labels),
    y=y_train_labels
)

class_weight_dict = dict(enumerate(class_weights))


class_weight_dict[6] *= 2.0
class_weight_dict[0] *= 1.2

print("Calculated Class Weights:")
print(class_weight_dict)

# %% COMPILE MODEL
model.compile(

    optimizer=Adam(learning_rate=0.001),

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)

# %% Early Stopping
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=0.00001,
    verbose=1
)
# %% MODEL TRAINING
history = model.fit(

    datagen.flow(
        x_train,
        y_train,
        batch_size=128
    ),
    epochs=40,
    validation_data=(x_test, y_test),
    verbose=1,
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weight_dict
)

# %% MODEL TESTING AND PERFORMANCE EVALUATION


y_pred = model.predict(x_test)

y_pred_class = np.argmax(y_pred, axis=1)

y_true = np.argmax(y_test, axis=1)

report = classification_report(
    y_true,
    y_pred_class,
    target_names=class_labels
)
print(report)

# %% CONFUSION MATRIX

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted Label')
plt.show()
# %% VISUALIZE TRAINING HISTORY

plt.figure()

plt.subplot(1, 2, 1)

plt.plot(history.history["loss"], label="Train Loss")

plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)

plt.plot(history.history["accuracy"], label="Train Accuracy")

plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#%% CONFUSION MATRIX


y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)


confusion_mtx = confusion_matrix(y_true, y_pred_classes)


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10,8))
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Where is the Error?')
plt.show()