from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class TinyImageNet(Dataset):
    def __init__(self, root: Path, split: str = "train", transform=None):
        assert split in ("train", "val"), "split must be 'train' or 'val'"
        self.root      = Path(root) / "tiny-imagenet-200"
        self.split     = split
        self.transform = transform

        with open(self.root / "wnids.txt") as f:
            self.wnids = sorted(line.strip() for line in f)
        self.wnid_to_idx = {w: i for i, w in enumerate(self.wnids)}

        self.wnid_to_name = {}
        with open(self.root / "words.txt") as f:
            for line in f:
                wnid, name = line.strip().split("\t", 1)
                self.wnid_to_name[wnid] = name.split(",")[0].strip()
        self.class_names = [self.wnid_to_name.get(w, w) for w in self.wnids]

        self.samples: list[tuple[Path, int]] = []
        if split == "train":
            for wnid in self.wnids:
                label   = self.wnid_to_idx[wnid]
                img_dir = self.root / "train" / wnid / "images"
                for img_file in sorted(img_dir.glob("*.JPEG")):
                    self.samples.append((img_file, label))
        else:
            ann_path = self.root / "val" / "val_annotations.txt"
            with open(ann_path) as f:
                for line in f:
                    parts    = line.strip().split("\t")
                    filename = parts[0]
                    wnid     = parts[1]
                    if wnid in self.wnid_to_idx:
                        img_path = self.root / "val" / "images" / filename
                        self.samples.append((img_path, self.wnid_to_idx[wnid]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
