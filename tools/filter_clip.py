import argparse
import json

import numpy as np
import torch
import clip
from PIL import Image

model_name = "ViT-B/32"
clip.available_models()

class Bbox_filter:
    def __init__(self):
        model, preprocess = clip.load("ViT-B/32")

        self.model = model
        self.preprocess = preprocess
        self.model.cuda().eval()

        self.labels = [
            [1, "a picture of people walking"], 
            [0, "a picture of stopping and straddling on a bike"], 
            [1, "a picture of walking along with a bike"],
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
        print(type(image))
        padding = 0
        pad_bboxes = list(map(
            lambda box: [
                max(0, box[0]-padding), 
                max(0, box[1]-padding), 
                min(image.width-1, box[2]+padding), 
                min(image.height-1, box[3]+padding)
            ], 
            bboxes
        ))
        images = []
        for bbox in pad_bboxes:
            print("bbox", bbox)
            images.append(self.preprocess(image.crop(bbox)))
        image_input = torch.tensor(np.stack(images)).cuda()
        image_features = self.model.encode_image(image_input).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ self.label_features.T).softmax(dim=-1)

        print("Text features:", self.label_features)
        print("Image features:", image_features)

        filtered_bboxes = []
        for prob, bbox in zip(similarity, bboxes):
            lab_id = torch.argmax(prob)
            print("Debug", prob, bbox)
            if self.labels[lab_id][0] == 1:
                filtered_bboxes.append(bbox)

        return similarity, filtered_bboxes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CLIP')
    parser.add_argument('--image', type=str)
    parser.add_argument('--bboxes', type=str)

    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")

    with open(args.bboxes, "r") as fi:
        bboxes = json.load(fi)

    assert "type" in bboxes
    if bboxes["type"] == "XYWH":
        bboxes["data"] = list(map(lambda x: [x[0], x[1], x[0]+x[2], x[1]+x[3]], bboxes["data"]))
    else:
        assert bboxes["type"] == "XYUV"

    filter = Bbox_filter()
    similarity, filtered = filter(image, bboxes["data"])

    print("Similarity", similarity)
    print("Filtered", filtered)
    