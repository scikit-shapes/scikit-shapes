class Dataset:

    def __init__(self, *, shapes=None, labels=None, landmarks=None, affine=None, **kwargs) -> None:
        
        self.shapes = shapes
        self.labels = labels
        self.landmarks = landmarks
        self.affine = affine


    def __len__(self):
        return len(self.shapes)
    
