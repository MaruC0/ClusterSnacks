import os
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

MODEL_FILE = "simclr_model.pt"
SIMCLR_IMG_SIZE = 64


# ── Neural Network Components ────────────────────────────────────

class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class _SmallCNN(nn.Module):
    """
    Backbone léger ~480K paramètres, adapté pour ~1000 images.
    Architecture :
        Conv block 1:  3 → 32   (64×64 → 32×32)
        Conv block 2:  32 → 64  (32×32 → 16×16)
        Conv block 3:  64 → 128 (16×16 → 8×8)
        Conv block 4:  128 → 256 (8×8 → 4×4)
        GlobalAvgPool → 256-d embedding
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            _ConvBlock(3, 32),
            _ConvBlock(32, 64),
            _ConvBlock(64, 128),
            _ConvBlock(128, 256),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.embed_dim = 256

    def forward(self, x):
        return self.encoder(x)


class _SimCLRNet(nn.Module):
    """Encoder + projection head pour l'entraînement contrastif."""
    def __init__(self):
        super().__init__()
        self.encoder = _SmallCNN()
        backbone_dim = self.encoder.embed_dim

        self.projector = nn.Sequential(
            nn.Linear(backbone_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
        )

    def forward(self, x):
        h = self.encoder(x)    # 256-d backbone embedding  — utilisé pour le clustering
        z = self.projector(h)  # 128-d projected embedding — utilisé pour l'entraînement
        return h, z


class _NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temp = temperature

    def forward(self, z1, z2):
        B = z1.size(0)
        z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
        sim = torch.mm(z, z.T) / self.temp

        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float('-inf'))

        labels = torch.cat([torch.arange(B, 2 * B), torch.arange(B)]).to(z.device)
        return F.cross_entropy(sim, labels)


# ── Datasets ─────────────────────────────────────────────────────

class _SimCLRDataset(Dataset):
    """Retourne deux vues augmentées de chaque image pour l'entraînement contrastif."""
    def __init__(self, image_cache, image_paths, image_size):
        self.cache = image_cache
        self.paths = image_paths
        self.image_size = image_size
        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([
                T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.15)
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=5)], p=0.5),
            T.RandomRotation(degrees=15),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = self.cache[idx]
        if img is None:
            blank = torch.zeros(3, self.image_size, self.image_size)
            return blank, blank
        return self.transform(img), self.transform(img)


class _InferenceDataset(Dataset):
    """Resize + normalize uniquement — pas d'augmentation."""
    def __init__(self, image_cache, image_paths, image_size):
        self.cache = image_cache
        self.paths = image_paths
        self.image_size = image_size
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = self.cache[idx]
        if img is None:
            return torch.zeros(3, self.image_size, self.image_size), self.paths[idx]
        return self.transform(img), self.paths[idx]


# ── Interface publique ───────────────────────────────────────────

class SimCLRModel:
    """
    Modèle SimCLR : CNN léger (~480K params) entraîné de zéro par
    apprentissage contrastif (NT-Xent loss). Produit des embeddings 256-dim.
    Pas de poids pré-entraînés — le réseau apprend directement sur vos images.
    """

    def __init__(self, img_size=SIMCLR_IMG_SIZE, models_dir="trained-models"):
        self.img_size = img_size
        self.models_dir = models_dir
        self.model = None
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

    def _build(self):
        self.model = _SimCLRNet().to(self.device)

    @staticmethod
    def _load_pil_rgb(path):
        """Charge une image et la convertit en RGB (gère RGBA, palette, grayscale)."""
        try:
            img = Image.open(path)
            if img.mode == "P":
                img = img.convert("RGBA")
            if img.mode == "RGBA":
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                img = bg
            elif img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except Exception:
            return None

    def _preload_images(self, image_paths):
        cache, skipped = [], 0
        for path in image_paths:
            img = self._load_pil_rgb(path)
            if img is None:
                skipped += 1
            cache.append(img)
        loaded = len(image_paths) - skipped
        print(f"[SimCLR] {loaded} images chargées en mémoire ({skipped} ignorées)")
        return cache

    # ── Entraînement ──────────────────────────────────────────────

    def train(self, image_paths, epochs=300, batch_size=32, learning_rate=3e-4, callback=None):
        self._build()

        image_cache = self._preload_images(image_paths)
        dataset = _SimCLRDataset(image_cache, image_paths, self.img_size)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, drop_last=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = _NTXentLoss(temperature=0.5)

        self.model.train()
        print(f"[SimCLR] Entraînement : {epochs} epochs, batch={batch_size}, device={self.device}")

        for epoch in range(epochs):
            epoch_loss = 0.0
            for v1, v2 in loader:
                v1, v2 = v1.to(self.device), v2.to(self.device)
                _, z1 = self.model(v1)
                _, z2 = self.model(v2)
                loss = criterion(z1, z2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(len(loader), 1)
            scheduler.step()

            print(f"  Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.4f}")
            if callback:
                callback(epoch + 1, epochs, avg_loss)

        self.save()

    # ── Sauvegarde / Chargement ───────────────────────────────────

    def save(self):
        os.makedirs(self.models_dir, exist_ok=True)
        path = os.path.join(self.models_dir, MODEL_FILE)
        torch.save(self.model.state_dict(), path)
        print(f"[SimCLR] Modèle sauvegardé : {path}")

    def load(self):
        path = os.path.join(self.models_dir, MODEL_FILE)
        if os.path.exists(path):
            self._build()
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
            print(f"[SimCLR] Modèle chargé : {path}")
            return True
        return False

    def is_trained(self):
        return os.path.exists(os.path.join(self.models_dir, MODEL_FILE))

    # ── Extraction de features ────────────────────────────────────

    def extract_features(self, img_paths, batch_size=32):
        """Extrait les features SimCLR (256-dim) pour une liste de chemins d'images."""
        if self.model is None:
            if not self.load():
                raise FileNotFoundError(
                    f"Modèle SimCLR non trouvé dans '{self.models_dir}'. "
                    "Veuillez d'abord entraîner le modèle."
                )

        self.model.eval()
        image_cache = self._preload_images(img_paths)
        dataset = _InferenceDataset(image_cache, img_paths, self.img_size)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        feats = []
        with torch.no_grad():
            for images, _ in loader:
                h, _ = self.model(images.to(self.device))
                feats.append(h.cpu().numpy())

        return np.vstack(feats) if feats else np.empty((0, 256))

    def extract_features_from_array(self, img_bgr):
        """Extrait les features SimCLR pour une seule image BGR (numpy array)."""
        if self.model is None:
            if not self.load():
                raise FileNotFoundError("Modèle SimCLR non trouvé.")

        self.model.eval()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        transform = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        tensor = transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            h, _ = self.model(tensor)
        return h.cpu().numpy()[0]
