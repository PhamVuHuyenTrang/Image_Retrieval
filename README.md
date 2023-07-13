# Image_Retrieval
Computer Vision Capstone Project 20222\
In this project, we build an efficient and accurate retrieval system of fashion products, utilizing Computer Vision and Deep Learning techniques.\
**Stage1** To train the bi-encoder model, please run image-retrieval-pipeline.ipynb
**Stage2 (Reranking)**: We ultilize three type of backborn: Vision Transformer, EfficientNet and ShuffleNet to train the cross-encoder model. This model is used to rescore the similarity between query image and each image in the shop, then improve the performance of bi-encoder model.\
**Object Detection** : We use YOLOv5 to detect fashion items in an image so that consumers can choose the item they desire to get similar images. The code for object detection training and demo can be found in https://github.com/PhamVuHuyenTrang/Image_Retrieval/tree/main/Object_Detection.
**Demo** To run the system, please place the deepfashion dataset in the same directory with the demo folder, and run all cells in demo.ipynb to load dataset with index and gradio interface
