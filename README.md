Current Hyperspectral Image Classification (HSIC) methods based on Transformer architectures have limitations in capturing local spatial-semantic features and often
ignore the inherent spectral-mixing phenomenon. These issues limit the models’ ability to fully utilize joint spatial-spectral information, leading to suboptimal classification
performance, especially under few-shot conditions. To address these challenges, we propose a multitask Transformer framework for HSIC, called Spatial–Spectral
Multiscale Tokenization Unmixing Transformer with Super Principal Component Analysis, which incorporates Super Principal Component Analysis (SuperPCA) to
improve feature representation. Specifically, to enhance the Transformer’s ability to capture spectral features, we introduce a spectral-attention block that enables dual
attention mechanisms. Meanwhile, we apply a Spatial-Spectral Multiscale Tokenization Transformer (SSMTT) to extract local features across different spatial scales. We then
develop a DenseNet-based, Dual-Branch Symmetric Unmixing (DBSU) module to generate an abundance map, serving as subpixel prior information for classification.
Finally, the classification map obtained by combining SSMTT and DBSU provides complementary information to the SuperPCA-denoised classification map. This
complementary information is further integrated through a secondary decision fusion module, significantly enhancing classification accuracy, particularly in few-shot
conditions. Experiments on four real datasets show that our framework outperforms the current best method by 0.08%, 0.02%, 0.64%, and 0.14%, while maintaining a runtime of 8.09 seconds the Indian Pines dataset. Our framework can be applied to agricultural land cover classification.(publication in EAAI 2026)
 This is the source code of the paper"SuperPCA with Spatial-Spectral Multiscale Tokenization and Unmixing Transformer for Hyperspectral Image Classification". Welcome to reproduce it and provide your feedback.
All data can be download from:
Link: https://pan.baidu.com/s/1OSK1eR_3OlUakDena3oFjA?pwd=1234
Code: 1234
Email:xuanliu2612@163.com
Code execution steps(Taking Indian Pines as an example):
(1) Running runFastHyIn.m to get figdata.mat.
(2) Running MSuperPCA.m and IP_train.py to get the final classification accuracy.
