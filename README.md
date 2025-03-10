# 🖼️ Image Retrieval

In this project, we build an efficient and accurate retrieval system of fashion products, utilizing Computer Vision and Deep Learning techniques.

## 🚀 Pipeline

### 🏁 Stage 1: Bi-Encoder Training  
📌 Run `image-retrieval-pipeline.ipynb` to train the bi-encoder model.

### 🎯 Stage 2: Reranking  
📌 We utilize three backbones:
- 🟡 Vision Transformer (ViT)
- 🟢 EfficientNet
- 🔵 ShuffleNet  
📌 These are used to train the cross-encoder model, which rescales the similarity between the query image and each shop image to enhance the bi-encoder performance.

### 🔍 Object Detection  
📌 YOLOv5 is used to detect fashion items in an image. Consumers can select an item to find similar images.  
📌 Object detection training and demo code: [Object Detection Repo](https://github.com/PhamVuHuyenTrang/Image_Retrieval/tree/main/Object_Detection)

## 🎮 Demo
📌 To run the system:
1. Place the DeepFashion dataset in the same directory as the `demo` folder.
2. Run all cells in `demo.ipynb` to load the dataset with index and Gradio interface.

---

## 👥 Contributors
This work was done as part of the **Computer Vision Course - IT4343E** at **Hanoi University of Science and Technology**. Team members include:

- 👨‍💻 [Đỗ Tuấn Anh](https://github.com/AnhDt-dsai)
- 👨‍💻 [Trần Xuân Huy](https://github.com/TranXuanHuy267)
- 👩‍💻 [Phạm Vũ Huyền Trang](https://github.com/PhamVuHuyenTrang)
- 👨‍💻 [Đào Trọng Việt](https://github.com/viet-data)
