import numpy as np
import torch
import clip
import from PIL import Image

model_name = "ViT-B/32"
clip.available_models()

class bbox_filter:
    def __init__(self):
        model, preprocess = clip.load("ViT-B/32")

        self.model = model
        self.preprocess = preprocess
        self.model.cuda().eval()

        self.labels = [
            [1, "a picture of people walking"], 
            [0, "a picture of stopping and straddling on a bike"], 
            [0, "a picture of walking along with a bike"],
            [0, "a picture of people on a vehicle"]
        ]

        self.label_tokens = clip.tokenize([lab[1] for lab in self.labels]).cuda()

        with torch.no_grad():
            self.label_features = self.model.encode_text(self.label_tokens).float()
            self.label_features /= self.label_features.norm(dim=-1, keepdim=True)

        input_resolution = model.visual.input_resolution
        context_length = model.context_length
        vocab_size = model.vocab_size

        print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
        print("Input resolution:", input_resolution)
        print("Context length:", context_length)
        print("Vocab size:", vocab_size)

    def __call__(self, image: Image, bboxes):
        images = []
        for bbox in bboxes:
            images.append(self.preprocess(image.crop(bbox)))
        image_input = torch.tensor(np.stack(images)).cuda()
        image_features = self.model.encode_image(image_input).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ self.label_features.T).softmax(dim=-1)
        
        print(similarity)

