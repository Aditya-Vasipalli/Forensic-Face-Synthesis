import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SketchPhotoDataset(Dataset):

    def __init__(self, sketch_dir, photo_dir, img_size=256):

        self.sketch_dir = sketch_dir
        self.photo_dir = photo_dir

        photo_files = set(os.listdir(photo_dir))
        all_sketches = sorted(f for f in os.listdir(sketch_dir) if f.endswith(".jpg"))

        # Only keep sketches whose matching photo actually exists
        self.files = [f for f in all_sketches if self._sketch_to_photo(f) in photo_files]
        skipped = len(all_sketches) - len(self.files)

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1, 1]
        ])

        print(f"Dataset: {len(self.files)} pairs loaded ({skipped} sketches skipped — no matching photo)")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        sketch_file = self.files[idx]
        photo_file = self._sketch_to_photo(sketch_file)

        sketch_path = os.path.join(self.sketch_dir, sketch_file)
        photo_path = os.path.join(self.photo_dir, photo_file)

        sketch = Image.open(sketch_path).convert("RGB")
        photo = Image.open(photo_path).convert("RGB")

        sketch = self.transform(sketch)
        photo = self.transform(photo)

        return sketch, photo

    @staticmethod
    def _sketch_to_photo(sketch_name):
        """Map sketch filename to photo filename.
        Handles artist prefixes: F2->f, M2->m, then strips -sz1."""
        name = sketch_name
        if name.startswith("F2-"):
            name = "f-" + name[3:]
        elif name.startswith("M2-"):
            name = "m-" + name[3:]
        return name.replace("-sz1", "")