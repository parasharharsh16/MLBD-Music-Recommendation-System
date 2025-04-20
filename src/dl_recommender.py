import logging
import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from config import AE_PARAMS, MODE, KNN_K
from .model_persistence import load_torch, save_torch, load_numpy, save_numpy

logger = logging.getLogger(__name__)

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim,16), nn.ReLU(), nn.Linear(16,latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim,16), nn.ReLU(), nn.Linear(16,input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

class DLRecommender:
    def __init__(self, audio_mat: np.ndarray, df):
        logging.getLogger(__name__).info('Initializing DLRecommender (mode=%s)', MODE)
        from .model_persistence import MODELS_DIR
        if MODE == 'train':
            import shutil, os
            shutil.rmtree(MODELS_DIR, ignore_errors=True)
            os.makedirs(MODELS_DIR, exist_ok=True)

        model = load_torch(Autoencoder, 'autoencoder.pth', input_dim=audio_mat.shape[1], latent_dim=AE_PARAMS['latent_dim'])
        emb = load_numpy('dl_embeddings')
        X = torch.tensor(audio_mat, dtype=torch.float32)
        if model is None or emb is None:
            model = Autoencoder(audio_mat.shape[1], AE_PARAMS['latent_dim'])
            opt = torch.optim.Adam(model.parameters(), lr=AE_PARAMS['lr'])
            loss_fn = nn.MSELoss()
            for _ in tqdm(range(AE_PARAMS['epochs']), desc='AE Training'):
                opt.zero_grad()
                recon = model(X)
                loss = loss_fn(recon, X)
                loss.backward()
                opt.step()
            save_torch(model, 'autoencoder.pth')
            with torch.no_grad():
                emb = model.encoder(X).numpy()
            save_numpy(emb, 'dl_embeddings')

        self.nn = NearestNeighbors(metric='euclidean')
        self.nn.fit(emb)
        self.emb = emb
        self.df = df

    # def recommend(self, q: str, k: int = KNN_K) -> list:
    #     mask = self.df['track_name'].str.contains(q.lower().strip(), case=False, na=False)
    #     if not mask.any():
    #         return self.df.sample(k)['track_name'].tolist()
    #     i = mask.idxmax()
    #     _, neigh = self.nn.kneighbors(self.emb[i].reshape(1, -1), n_neighbors=k+1)
    #     return self.df.iloc[neigh[0][1:]]['track_name'].tolist()
    def recommend(self, q: str, k: int = KNN_K) -> list[tuple[str, float]]:
        mask = self.df['track_name'].str.contains(q.lower().strip(), case=False, na=False)
        
        if not mask.any():
            # Return k random samples with score 0.0
            random_tracks = self.df.sample(k)['track_name'].tolist()
            return [(track, 0.0) for track in random_tracks]

        i = mask.idxmax()
        dists, neigh = self.nn.kneighbors(self.emb[i].reshape(1, -1), n_neighbors=k+1)
        
        # Skip the first neighbor (the track itself), and return (track_name, similarity_score)
        neighbors = neigh[0][1:]
        distances = dists[0][1:]
        scores = 1 / (1 + distances)  # Convert distance to similarity score (range: (0,1])

        return [(self.df.iloc[idx]['track_name'], float(score)) for idx, score in zip(neighbors, scores)]
