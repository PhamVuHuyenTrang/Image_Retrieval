
from huggingface_hub import hf_hub_download
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torch
from transformers import ViTImageProcessor, ViTModel
import torchvision.transforms as T


class Initial:
    def __init__(self, model="ShuffleNet"):
        assert model in ['ViT', "ShuffleNet", "EfficientNet"]
        if model == "ShuffleNet":
            self.process = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.model = rerankingShuffleNet()
            version = None
            path = hf_hub_download(repo_id="Huy1432884/rerankingShuffleNet",
                filename="model.bin", 
                use_auth_token="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", 
                #local_dir="/kaggle/working/",
                revision=version
            )
        elif model == "ViT":
            self.process = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
            self.topil = T.ToPILImage()
            self.model = rerankingViT()
            version = None
            path = hf_hub_download(repo_id="Huy1432884/rerankingViT", 
                filename="model.bin", 
                use_auth_token="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", 
                #local_dir="/kaggle/working/",
                revision=version
            )
        elif model=="EfficientNet":
            self.process = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.model = rerankingEfficientNet()
            version = None
            path = hf_hub_download(repo_id="Huy1432884/rerankingEfficientnet",
                filename="model.bin", 
                use_auth_token="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", 
                #local_dir="/kaggle/working/",
                revision=version
            )

        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self.model.eval()
        
def reranking(image1, image2s, initial, model="ShuffleNet"):
    assert model in ['ViT', "ShuffleNet", "EfficientNet"]
    if model == 'ViT':
        IMG_SIZE=224
    else:
        IMG_SIZE=256

    def merge_height(image1, image2):
        new_image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (250, 250, 250))
        new_image.paste(image1,(0,0))
        new_image.paste(image2,(int(IMG_SIZE)//2,0))
        return new_image
    
    
    img1 = transforms.Resize((IMG_SIZE, int(IMG_SIZE/2)))(image1)
    merged_imgs = []
    for image2 in image2s:
        img2 = transforms.Resize((IMG_SIZE, int(IMG_SIZE/2)))(image2)
        merged_img = merge_height(img1, img2)
    
        if model == 'ShuffleNet' or model == "EfficientNet":
            merged_img = initial.process(merged_img).unsqueeze(0)
        elif model == "ViT":
            trans = transforms.ToTensor()
            merged_img = trans(merged_img)
            merged_img = initial.process(initial.topil(merged_img), return_tensors="pt")["pixel_values"]
        merged_imgs.append(merged_img)
    merged_imgs = torch.cat(merged_imgs, dim=0)
    output = initial.model(merged_imgs)
    output = output.squeeze(1).tolist()
    return output
    
class rerankingShuffleNet(nn.Module):
    def __init__(self):
        super().__init__()
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', pretrained=False)
        self.model.fc = nn.Linear(1024, 1)
        self.act = nn.Sigmoid()
        self.crit = nn.MSELoss()
    
    def forward(self, merged_imgs):
        logits = self.model(merged_imgs)
        output = self.act(logits)
        return output

class rerankingViT(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.extractor = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.hidden_size = self.extractor.config.hidden_size
        self.fc = nn.Linear(768, 1)
        self.act = nn.Sigmoid()
        self.crit = nn.MSELoss()
    
    def forward(self, merged_imgs):
        embed = self.extractor(merged_imgs)[0].mean(dim=1)
        logits = self.fc(embed)
        output = self.act(logits)
        return output

class rerankingEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=False)
        self.model.classifier.fc = nn.Linear(1280, 1)
        self.act = nn.Sigmoid()
        self.crit = nn.MSELoss()
    
    def forward(self, merged_imgs):
        logits = self.model(merged_imgs)
        output = self.act(logits)
        return output
