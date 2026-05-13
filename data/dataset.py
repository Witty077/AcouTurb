import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class AcouTurbDataset(Dataset):
    """
    Dataset for anomalous signal detection (AcouTurb).
    Handles loading real audio/signal data or generating synthetic mock data
    for development when real data is unavailable.
    """
    def __init__(
        self,
        data_dir=None,
        sampling_rate=16000,
        duration=2.0,
        mode="train",
        transform=None,
        use_mock_data=True,
        mock_num=100
    ):
        """
        Args:
            data_dir (str, optional): Path to the directory containing 'train' and 'test' subfolders.
            sampling_rate (int): Target sample rate.
            duration (float): Duration of each signal snippet in seconds.
            mode (str): 'train' or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
            use_mock_data (bool): If True and no valid data is found, generates synthetic data.
            mock_num (int): Number of mock samples to generate if use_mock_data is True.
        """
        self.data_dir = data_dir
        self.file_paths = []
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.mode = mode
        self.transform = transform
        self.num_sampling = int(self.sampling_rate * self.duration)
        self.labels = [] # 0 for normal, 1 for anomalous
        self.use_mock_data = use_mock_data

        if self.data_dir and os.path.exists(self.data_dir):
            target_dir = os.path.join(self.data_dir, self.mode)
            if os.path.exists(target_dir):
                wav_files = glob.glob(os.path.join(target_dir, "*.wav"))
                self.file_paths.extend(wav_files)
                
                for f in self.file_paths:
                    filename = os.path.basename(f).lower()
                    if self.mode == "train":
                        # In train mode, all data is considered normal
                        self.labels.append(0)
                    elif self.mode == "test":
                        # In test mode, determine label from filename
                        if "abnormal" in filename:
                            self.labels.append(1)
                        else:
                            # Defaulting everything else to normal
                            self.labels.append(0)

        else:
            if self.use_mock_data:
                print("No valid data_dir provided. Using synthetic mock data for development.")
                self.mock_num = mock_num
                if self.mode == "train":
                    self.labels = [0] * self.mock_num
                else:
                    # Test contains both normal and anomalous samples
                    self.labels = np.random.randint(0, 2, size=self.mock_num).tolist()
            else:
                raise ValueError("No data_dir provided and use_mock_data is False.")
            

    def __len__(self):
        if self.use_mock_data:
            return self.mock_num
        return len(self.file_paths)

    def __getitem__(self, idx):
        if self.use_mock_data:
            # Generate random white noise as a mock signal
            signal = np.random.randn(self.num_sampling).astype(np.float32)
            label = self.labels[idx]
            
            # Add a specific pattern (e.g., sine wave) if it's anomalous
            if label == 1:
                t = np.linspace(0, self.duration, self.num_sampling)
                signal += 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
                
            signal_tensor = torch.tensor(signal)
            
        else:
            label = self.labels[idx]
            
            # In a real scenario, use torchaudio or librosa to load the audio:
            # import torchaudio
            # signal_tensor, sr = torchaudio.load(file_path)
            # if sr != self.sampling_rate:
            #     resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)
            #     signal_tensor = resampler(signal_tensor)
            # For now, we mock the loading to avoid strict dependency failures if not installed yet.
            signal_tensor = torch.randn(self.num_sampling, dtype=torch.float32)

        # Standardize shape to [Channels, Time] -> [1, num_sampling]
        if signal_tensor.dim() == 1:
            signal_tensor = signal_tensor.unsqueeze(0)

        if self.transform:
            signal_tensor = self.transform(signal_tensor)

        return signal_tensor, torch.tensor(label, dtype=torch.long)

# Example usage for testing the dataset implementation
if __name__ == "__main__":
    print("Testing AcouTurbDataset with mock train data...")
    train_dataset = AcouTurbDataset(mode="train", use_mock_data=True, mock_num=512)
    for i in range(len(train_dataset)):
        signal, label = train_dataset[i]
        print(f"Train Sample {i}: Shape={signal.shape}, Label={label.item()}")

    print("\nTesting AcouTurbDataset with mock test data...")
    test_dataset = AcouTurbDataset(mode="test", use_mock_data=True, mock_num=256)
    for i in range(len(test_dataset)):
        signal, label = test_dataset[i]
        print(f"Test Sample {i}: Shape={signal.shape}, Label={label.item()}")
