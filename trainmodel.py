from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
label_map = {label:num for num, label in enumerate(actions)}
# print(label_map)
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        missing = False
        for frame_num in range(sequence_length):
            npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            if not os.path.isfile(npy_path):
                print(f"Warning: missing file {npy_path}, skipping sequence {action}/{sequence}")
                missing = True
                break
            try:
                res = np.load(npy_path)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"Warning: failed to load {npy_path} ({e}), skipping sequence {action}/{sequence}")
                missing = True
                break
            window.append(res)
        if not missing:
            sequences.append(window)
            labels.append(label_map[action])
    # Progress update per action
    print(f"Processed action {action}: collected {sum(1 for l in labels if l==label_map[action])} sequences so far")

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
res = [.7, 0.2, 0.1]

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
history = model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])
model.summary()

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')

# Save training history to analysis/ for later plotting
try:
    import json
    analysis_dir = os.path.join(os.getcwd(), 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    hist_path = os.path.join(analysis_dir, 'training_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history.history, f)
    print(f"Saved training history to {hist_path}")
except Exception as e:
    print('Could not save training history:', e)