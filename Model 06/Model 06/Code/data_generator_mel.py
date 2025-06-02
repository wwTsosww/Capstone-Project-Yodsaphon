
import os
import numpy as np
import tensorflow as tf

class MelDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, input_root_dir, target_root_dir, batch_size=8, shuffle=True):
        self.input_root_dir = input_root_dir
        self.target_root_dir = target_root_dir
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.input_paths = self._load_file_paths(self.input_root_dir)
        self.target_paths = self._load_file_paths(self.target_root_dir)

        assert len(self.input_paths) == len(self.target_paths), "Mismatch between input and target samples!"

        self.indexes = np.arange(len(self.input_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _load_file_paths(self, root_dir):
        paths = []
        for song_folder in os.listdir(root_dir):
            song_mix_folder = os.path.join(root_dir, song_folder, "mix")
            if not os.path.exists(song_mix_folder):
                continue
            for fname in sorted(os.listdir(song_mix_folder)):
                if fname.endswith(".npy"):
                    full_path = os.path.join(song_mix_folder, fname)
                    paths.append(full_path)
        return paths

    def __len__(self):
        return int(np.ceil(len(self.input_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_input_files = [self.input_paths[k] for k in batch_indexes]
        batch_target_files = [self._get_target_path_from_input_path(p) for p in batch_input_files]

        X, Y = self.__data_generation(batch_input_files, batch_target_files)

        return X, Y

    def _get_target_path_from_input_path(self, input_path):
        song_name = input_path.split(os.sep)[-3]
        segment_name = os.path.basename(input_path)

        target_paths = [
            os.path.join(self.target_root_dir, song_name, "stems", "vocals", segment_name),
            os.path.join(self.target_root_dir, song_name, "stems", "drums", segment_name),
            os.path.join(self.target_root_dir, song_name, "stems", "bass", segment_name),
            os.path.join(self.target_root_dir, song_name, "stems", "other", segment_name),
        ]

        return target_paths

    def __data_generation(self, batch_input_files, batch_target_files):
        X_batch = []
        Y_batch = []

        for input_file, target_files in zip(batch_input_files, batch_target_files):
            x = np.load(input_file)
            if x.shape[1] < 864:
                pad_width = 864 - x.shape[1]
                x = np.pad(x, ((0, 0), (0, pad_width)), mode='constant')
            x = x[:, :864]
            x = np.expand_dims(x, axis=-1)

            y_list = []
            for tf_path in target_files:
                y = np.load(tf_path)
                if y.shape[1] < 864:
                    pad_width = 864 - y.shape[1]
                    y = np.pad(y, ((0, 0), (0, pad_width)), mode='constant')
                y = y[:, :864]
                y_list.append(y)

            y = np.stack(y_list, axis=-1)
            X_batch.append(x)
            Y_batch.append(y)

        X_batch = np.array(X_batch)
        Y_batch = np.array(Y_batch)

        return X_batch, Y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
