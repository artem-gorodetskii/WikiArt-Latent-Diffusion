import torch
import pickle
from torch.utils.data import Dataset


class WikiArtDataset(Dataset):
    r"""Dataset class.
     """
    def __init__(self, samples_paths):
        self.samples_paths = samples_paths

    def __len__(self):
        return len(self.samples_paths)

    def __getitem__(self, idx):

        with open(self.samples_paths[idx], 'rb') as handle:
          data = pickle.load(handle)

        image = data['latent']
        artist_label = data['artist']
        genre_label = data['genre']
        style_label = data['style']

        image = torch.FloatTensor(image)

        artist_label = torch.LongTensor([artist_label])
        genre_label = torch.LongTensor([genre_label])
        style_label = torch.LongTensor([style_label])

        return image, artist_label, genre_label, style_label
