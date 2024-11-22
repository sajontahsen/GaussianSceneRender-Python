> A simplified Python implementation of **3D Gaussian Splatting (3DGS)** scene rendering. This initializes the Gaussians from COLMAP input and renders a scene based on the original 3DGS implementation. The idea is to learn the core concepts of Gaussian Splatting by building up a barebones version from scratch. Lots more to be added!
---


### **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/GaussianSceneRender-Python.git
   cd GaussianSceneRender-Python
   ```

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

### **Usage**

To render an image from COLMAP reconstruction data, you can use the command-line interface:
```bash
python main.py --colmap_path <path_to_colmap_data> --image_id <image_id>
```

Example:
```bash
python main.py --colmap_path "treehill/sparse/0" --image_id 100
```

This will render the specified image and save the output to `./output/rendered_image_<image_id>.png`

---

### **References**
- [Original 3D Gaussian Splatting Repo](https://github.com/graphdeco-inria/gaussian-splatting/tree/main)
- [COLMAP Read/Write Model Script](https://github.com/colmap/colmap/blob/main/scripts/python/read_write_model.py)
- [An Introduction to 3D Gaussian Splatting (Article)](https://towardsdatascience.com/a-python-engineers-introduction-to-3d-gaussian-splatting-part-1-e133b0449fc6)

