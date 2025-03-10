# Image_Retrieval
In this project, we build an efficient and accurate retrieval system of fashion products, utilizing Computer Vision and Deep Learning techniques.\
**Stage 1** Run `image-retrieval-pipeline.ipynb` to train the bi-encoder model \
**Stage 2 (Reranking)**: We utilize three backbones: Vision Transformer, EfficientNet and ShuffleNet to train the cross-encoder model. This model is used to rescore the similarity between query image and each image in the shop, then improve the performance of the bi-encoder model.\
**Object Detection**: We use YOLOv5 to detect fashion items in an image so that consumers can choose the item they desire to get similar images. The code for object detection training and demo can be found in https://github.com/PhamVuHuyenTrang/Image_Retrieval/tree/main/Object_Detection.
**Demo** To run the system, please place the deepfashion dataset in the same directory with the demo folder, and run all cells in demo.ipynb to load the dataset with index and gradio interface.

## Contributors

This work was done as part of the `Computer Vision Course-IT4343E` at Hanoi University of Science and Technology. Team members include:

- [Đỗ Tuấn Anh](https://github.com/AnhDt-dsai)
- [Trần Xuân Huy](https://github.com/TranXuanHuy267)
- [Phạm Vũ Huyền Trang](https://github.com/PhamVuHuyenTrang)
- [Đào Trọng Việt](https://github.com/viet-data)

