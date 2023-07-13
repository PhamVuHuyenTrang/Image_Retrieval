from token import tok_name
from transformers import ViTImageProcessor, ViTForImageClassification, ViTModel
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms as T
import torch.nn as nn
import os
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from datasets import Dataset
import pandas as pd
import faiss
from multiprocessing import Pool
from copy import deepcopy
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def read_image(paths):
    return [Image.open(path) for path in paths]

def multi_parse(dataset):
    num_processes = os.cpu_count()
    chunk = 2
    with Pool(processes=num_processes) as pool:
        results = []
        for page in range(len(dataset)//chunk+1):
            results.append(pool.apply_async(read_image, args=(dataset[page*chunk:(page+1)*chunk],)))
        results = [i.get() for i in results]
        results = [j for i in results for j in i]
    return results

class generator(nn.Module):
    
    def __init__(self, hidden_size, outsize=256):
        super().__init__()
        channels = [hidden_size, 512, 256, 64, 32, 16, 8, 4]
        #          [2             4    8    16   32 64  128 512]
        self.gen = []
        self.init = 1
        self.hidden_size = hidden_size
        self.project = nn.Linear(hidden_size, hidden_size*self.init**2)
        for idx, i in enumerate(channels):
            inc = i
            if idx == len(channels)-1:
                outc = 3
            else:
                outc = channels[idx+1]
            self.gen.append(
                nn.Sequential(
                    nn.ConvTranspose2d(inc,
                                       outc,
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(outc),
                    nn.LeakyReLU())
            )
        self.gen = nn.Sequential(*self.gen)
        self.decode = nn.Sequential( nn.Conv2d(3, out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())
    def forward(self, hidden):
        hidden = self.project(hidden)
        hidden = hidden.reshape(-1,self.hidden_size,self.init, self.init)
        image = self.gen(hidden)
        image = self.decode(image)
        return image
    
class crossencoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.extractor = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.hidden_size = self.extractor.config.hidden_size
        self.gen = generator(self.hidden_size)
        self.resize = transforms.Resize(256)
        self.mu = nn.Linear(self.hidden_size, self.hidden_size)
        self.var = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.crit = nn.MSELoss()
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, img1, img2, labels, ori, img_gen):
        embed1 = self.extractor(img1)[0].mean(dim=1)
        embed2 = self.extractor(img2)[0].mean(dim=1)
        scores = self.score(torch.cat([embed1, embed2, embed1-embed2], dim=1))
        loss1 = self.crit(scores, labels)
        
        #vae
        img = self.test_gen(img_gen)
        loss2 = self.crit(img, self.resize(ori))
        
        return loss1+loss2
    
    def test_gen(self, img):
        embed = self.extractor(img)[0].mean(dim=1)
        hidden = self.reparameterize(self.mu(embed), self.var(embed))
        img = self.gen(hidden)
        return img
    
class bi_encoder:

    def __init__(self, revision=None):
        model = crossencoder()
        if revision is None:
            path =  hf_hub_download(repo_id="vietdata/crossencoder", filename="model.bin")
        else:
            path =  hf_hub_download(repo_id="vietdata/crossencoder", filename="model.bin", revision=revision)
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        extractor = model.extractor
        extractor.eval()
        self.extractor = extractor.to(device)
        self.process = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.embeddings_dataset = None
    
    def encode_dataset(self, loader):
        embeddings = []
        iters = tqdm(loader)
        paths = []
        for batch in iters:
            with torch.no_grad():
                embeddings.append(self.extractor(batch["img"].to(device))[0][:,0,:])#.mean(dim=1))
                paths.extend(batch["path"])
        embeddings = torch.cat(embeddings, dim=0)
        #norm embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings_dataset = Dataset.from_pandas(pd.DataFrame({"embeddings":[i.detach().cpu().numpy() for i in embeddings],
                                                    "paths":paths}))
        
        embeddings_dataset.add_faiss_index(column="embeddings", metric_type=faiss.METRIC_INNER_PRODUCT)

        self.embeddings_dataset = embeddings_dataset
    
    def get_embedding(self, images, close=True):
        topil = T.ToPILImage()
        def norm(image):
            #image = topil(torch.permute(torch.tensor(image),(2, 0,1)))
            height, width= image.height, image.width
            trans = transforms.Compose([
                transforms.CenterCrop(min(height, width)),
                transforms.Resize(224),
                transforms.ToTensor(),
            ])
            ori = image
            image = trans(image).unsqueeze(0)
            if image.size(1) == 1:
                image = image.repeat(1,3, 1, 1)
            return image
        lst_imgs = []
        for img in images:
            lst_imgs.append(topil(norm(img)[0]))

        imgs = self.process(lst_imgs, return_tensors="pt" )["pixel_values"]
        if close:
            for img in images:
                img.close()
        
        #return self.extractor(imgs.to(device))[0].mean(dim=1).cpu().detach().numpy()
        return F.normalize(self.extractor(imgs.to(device))[0][:, 0,:], p=2, dim=1).cpu().detach().numpy()
    
    def search_images_from_image(self, image, topk=100, close=True, label=None):
        query = self.get_embedding([image], close=close)
        #if label is not None:
        #    label = label.replace("_", "/")
        subdataset = self.embeddings_dataset
        if label is None:
            scores, samples = self.embeddings_dataset.get_nearest_examples(
            "embeddings", query, k=topk
        )
        else:
            scores, samples = [], {"paths":[]}
            for idx, i in enumerate(subdataset["paths"]):
                if label in i.replace("/", "_"):
                    samples["paths"].append(i)
                    scores.append(np.sum(query[0]*np.array(subdataset[idx]["embeddings"])))
            ndices = np.argsort(np.array(scores))
            samples["paths"] = [samples["paths"][i] for i in ndices[::-1][:topk]]
            scores = [scores[i] for i in ndices[::-1][:topk]]
        return [Image.open(os.path.join(i)) for i in samples["paths"]], scores

    def search_images_from_batch_image(self, imgs, topk=100, close=True):
        results = []
        queries = self.get_embedding(imgs, close=close)
        used = {}
        for query in queries:
            scores, samples = self.embeddings_dataset.get_nearest_examples(
                    "embeddings", query, k=topk
                )
            #print(samples["paths"])
            #for i in samples["paths"]:
            #    if i not in used:
            #        used[i] = Image.open(os.path.join(i))
            results.append((samples["paths"], scores))
        return results 

