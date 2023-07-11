
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import yolov5
import torch
from torchvision import transforms

class ObjectDetection:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model()
        self.model.eval()
        self.id2label = {
            0: 'TOPS_T_Shirt',
            1: 'TROUSERS_Summer_Suit',
            2: 'TOPS_Coat',
            3: 'TROUSERS_Pants',
            4: 'TOPS_Summer_Suit',
            5: 'DRESSES_Dress',
            6: 'TOPS_Tank_Top',
            7: 'TROUSERS_Leggings',
            8: 'CLOTHING_Pants',
            9: 'TOPS_Lace_Shirt',
            10: 'DRESSES_Skirt',
            11: 'CLOTHING_Blouse',
            12: 'TOPS_Chiffon',
            13: 'CLOTHING_T_Shirt',
            14: 'TOPS_Blouse',
            15: 'CLOTHING_Jeans',
            16: 'CLOTHING_Summer_Suit',
            17: 'CLOTHING_Polo_Shirt',
            18: 'DRESSES_Sleeveless_Dress',
            19: 'CLOTHING_Tank_Top',
            20: 'DRESSES_Lace_Dress',
            21: 'DRESSES_Suspenders_Skirt',
            22: 'CLOTHING_Coat',
        }

    def load_model(self):
        hf_hub_download(
            repo_id="Huy1432884/ODYolov5m",
            filename="model.pt",
            use_auth_token="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",
            local_dir="kaggle/working/",
            revision=None
        )
        model = yolov5.load("kaggle/working/model.pt")
        model.conf = 0.15
        model.iou = 0.45
        model.agnostic = False
        model.multi_label = False
        model.max_det = 1000
        model.to(self.device)
        return model

    def detect(self, img):
        results = self.model(img)
        img_array = np.array(img)
        predictions = results.pred[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        categories = predictions[:, 5]
        labels = []
        cropped_images = []
        cropped_image_arrays = []
        coors = []
        for i in range(len(predictions)):
            box = boxes[i]
            score = scores[i]
            category = categories[i]
            if score > self.model.conf:
                x1, y1, x2, y2 = box
                cropped_image_array = img_array[int(x1):int(x2), int(y1):int(y2), :]
                cropped_image = Image.fromarray(cropped_image_array)
                labels.append(self.id2label[int(category)])
                cropped_images.append(cropped_image)
                cropped_image_arrays.append(cropped_image_array)
                coors.append([x1, y1, x2, y2])
        return labels, cropped_images, cropped_image_arrays, coors
