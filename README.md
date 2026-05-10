# Equipment Fault Detection using Sensor Data Images
### Human Capital Digitalisation Initiative (NCDMB | 3Logy | Seplat)

## 📌 Problem Definition
In the oil and gas industry, manual inspection of equipment surface defects is time-consuming and prone to human error. This project leverages **Computer Vision** and **Deep Learning** to automate the detection of equipment faults. By converting industrial sensor data into visual representations, we can identify critical defects such as crazing, inclusion, patches, pitted-surface, rolled-in-scale and scratches in real-time to prevent catastrophic failures.

## 📊 Dataset Exploration
* **Data Type:** 2D images representing surface defects from industrial equipment.
* **Categories:** 6 distinct classes:
    * `crazing`
    * `inclusion`
    * `patches`
    * `pitted-surface`
    * `rolled-in_scale`
    * `scratches`
* **Input Features:** RGB images $(224 \times 224$ pixels).
* **Labels:** Categorical indices mapped via a JSON dictionary.

## 🛠 Data Preparation
To prepare the data for the CNN, the following steps were taken:
1.  **Resizing:** All images were resized to $224 \times 224$ to match the ResNet-18 input requirements.
2.  **Normalization:** Applied ImageNet statistics (Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]`) to ensure stable training.
3.  **Augmentation:** Used random horizontal flips and rotations to improve model generalization.

## 🧠 Model Design (CNN)
* **Architecture:** **ResNet-18** (Transfer Learning).
* **Why CNN?** Convolutional Neural Networks are designed to extract spatial hierarchies of features, making them perfect for identifying the complex textures found in surface defects.
* **Layers:** Utilized 18 deep layers including convolutional layers, batch normalization, and a final fully-connected (FC) layer adjusted for **6 classes**.

## 🚀 Deployment (Streamlit)
The model is deployed as a web application using **Streamlit**.
* **Input:** Users can upload a `.jpg` or `.png` image of an equipment surface.
* **Processing:** The backend uses PyTorch to run inference on the uploaded image.
* **Output:** The app displays the predicted Fault Category and the Confidence Score.

## ⚠️ Challenges & Improvements
* **Challenge:** Encountered a `state_dict` mismatch during deployment due to architectural differences between the training environment and the production script.
* **Solution:** Corrected the model blueprint to ResNet-18 and implemented `strict=False` loading to bridge the gap between training and inference.
* **Improvement:** Future iterations will include a "Heatmap" (Grad-CAM) to show exactly where the CNN "sees" the defect on the equipment.

## 💡 Reflection
This project taught me the end-to-end lifecycle of an AI product—from data exploration to solving real-world deployment bugs. This technology is vital for **Digital Transformation** in Nigeria's energy sector, providing indigenous capacity for automated predictive maintenance.

---
**Developed by:** Murjanatu Jaafaru Kamara 
**Program:** Human Capital Digitalisation Initiative  
**Tools:** PyTorch, Streamlit, Git, Python