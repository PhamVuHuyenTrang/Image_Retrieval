# 🚀 Image Retrieval for Fashion Products  

This project aims to build a **highly efficient and accurate** image retrieval system for fashion products, leveraging **Computer Vision** and **Deep Learning** techniques. By combining **bi-encoder retrieval**, **cross-encoder reranking**, and **object detection**, we significantly enhance search precision, enabling users to find visually similar fashion items effortlessly.  

## 🔍 Pipeline Overview  

### **Stage 1: Bi-Encoder Model**  
Train the bi-encoder model using the `image-retrieval-pipeline.ipynb` notebook. This model efficiently retrieves the top-k most relevant images based on feature embeddings.  

### **Stage 2: Reranking with Cross-Encoder**  
To improve retrieval accuracy, we employ a **cross-encoder model** using three powerful backbones:  
✔ **Vision Transformer (ViT)** – Captures global context for enhanced feature extraction.  
✔ **EfficientNet** – Optimized for performance with high accuracy and efficiency.  
✔ **ShuffleNet** – Lightweight and optimized for mobile applications.  

This reranking model **rescales similarity scores** between the query image and candidate images, refining the results provided by the bi-encoder.  

### **🎯 Object Detection: YOLOv5 for Fine-Grained Search**  
For **precise product search**, we integrate an object detection model (**YOLOv5**) to identify individual fashion items in an image. Users can then search for similar products based on detected objects.  

📌 **Object Detection Code**: [Object Detection Module](https://github.com/PhamVuHuyenTrang/Image_Retrieval/tree/main/Object_Detection)  

## 🛠️ Running the Demo  
To experience the image retrieval system:  

1️⃣ Place the **DeepFashion dataset** in the same directory as the `demo` folder.  
2️⃣ Open `demo.ipynb` and run all cells.  
3️⃣ The dataset will be indexed, and a **Gradio-powered UI** will be launched for easy interaction.  

## 👥 Contributors  

This project was developed as part of the **Computer Vision Course (IT4343E)** at **Hanoi University of Science and Technology**.  

- 🎓 [Đỗ Tuấn Anh](https://github.com/AnhDt-dsai)  
- 🎓 [Trần Xuân Huy](https://github.com/TranXuanHuy267)  
- 🎓 [Phạm Vũ Huyền Trang](https://github.com/PhamVuHuyenTrang)  
- 🎓 [Đào Trọng Việt](https://github.com/viet-data)  
