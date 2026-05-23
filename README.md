# Pseudo RGB-D SLAM — ROS 2 Package

> **Replacing a physical depth sensor with a neural network.**
> A modular ROS 2 pipeline that fuses monocular RGB from an IMX219 camera (or the TUM RGB-D dataset) with metric depth estimated by **Depth Anything V2** / **UniDepth**, feeding the result into **ORB-SLAM2** (RGB-D mode) for real-time visual odometry and 3-D mapping.

---

## Demo

### Live Camera real time (Jetson Orin Nano Super developer kit + IMX219)

| RViz2 — Depth Map & Trajectory | RViz2 — Map Points & Camera Pose |

| ![RViz2 Depth + Trajectory]<img width="1678" height="914" alt="Screenshot 2026-05-24 003136" src="https://github.com/user-attachments/assets/0ffae5b6-6cdf-4dc2-9ca6-21719830ecc7" />
### Visualizer Window (RGB · Depth · Trajectory — all simultaneous) with Dataset

![Visualizer]With dataset Visualization<img width="1176" height="679" alt="Screenshot from 2026-05-24 02-08-25" src="https://github.com/user-attachments/assets/5304365e-c44c-4448-9b1c-6ab09f94eac0" />
)


### Screen Recordings

| Mode | Link |
|---|---|
| 🎥 Live Camera (Jetson + IMX219) | [Watch on Google Drive](https://drive.google.com/your-link-here) |
| 🎥 TUM fr1/desk Dataset | [Watch on Google Drive](https://drive.google.com/your-link-here) |

> **To update:** replace the three `docs/images/` placeholders with your screenshots and update the Google Drive links above.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Jetson Orin Nano Super                        │
│                                                                   │
│  IMX219 CSI Camera                                                │
│    └──► nvarguscamerasrc (GStreamer HW NVMM)                     │
│              │                                                    │
│              ▼                                                    │
│  ┌─────────────────────────┐                                      │
│  │  Node A  · Broadcaster  │──► /camera/image_raw  (BGR 960×540) │
│  └─────────────────────────┘    /camera/camera_info              │
│              │                                                    │
│              ▼                                                    │
│  ┌─────────────────────────┐                                      │
│  │  Node B  · Depth Est.   │──► /depth/image_raw   (float32 m)  │
│  │  UniDepth / DepthAny V2 │    /depth/image_visual (colormap)  │
│  │  ViT-Small · FP16 CUDA  │                                     │
│  └─────────────────────────┘                                      │
│         │ RGB          │ Depth                                    │
│         └──────┬───────┘                                         │
│                ▼                                                  │
│  ┌─────────────────────────┐                                      │
│  │  Node C  · SLAM         │──► /slam/pose        (PoseStamped) │
│  │  ORB-SLAM2  RGB-D mode  │    /slam/trajectory  (Path)        │
│  │  ORB → PnP → BA → LC   │    /slam/map_points  (PointCloud2) │
│  └─────────────────────────┘                                      │
│                │                                                  │
│                ▼                                                  │
│  ┌─────────────────────────┐                                      │
│  │  Node D  · Visualizer   │──► OpenCV window                   │
│  │  RGB | Depth | Traj     │    (RGB feed + depth + trajectory)  │
│  └─────────────────────────┘                                      │
└──────────────────────────────────────────────────────────────────┘
```

### ROS Topics

| Topic | Type | Publisher → Subscriber |
|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` | A → B, C, D |
| `/camera/camera_info` | `sensor_msgs/CameraInfo` | A → B, C |
| `/depth/image_raw` | `sensor_msgs/Image` (float32 m) | B → C |
| `/depth/image_visual` | `sensor_msgs/Image` (BGR colormap) | B → D, RViz |
| `/slam/pose` | `geometry_msgs/PoseStamped` | C → D, RViz |
| `/slam/trajectory` | `nav_msgs/Path` | C → RViz |
| `/slam/map_points` | `sensor_msgs/PointCloud2` | C → RViz |
| `/slam/status` | `std_msgs/String` | C → terminal |

---

## Mathematical Background

### 1 · Pinhole Camera Model

A 3-D point **P** = [X, Y, Z]ᵀ projects to pixel **p** = [u, v]ᵀ via the intrinsic matrix **K**:

$$\begin{pmatrix}u\\v\\1\end{pmatrix} = \frac{1}{Z}\mathbf{K}\mathbf{P}, \qquad \mathbf{K}=\begin{pmatrix}f_x & 0 & c_x\\0 & f_y & c_y\\0 & 0 & 1\end{pmatrix}$$

Back-projection from depth map $D(u,v)$:

$$X = \frac{(u-c_x)\,D(u,v)}{f_x},\quad Y = \frac{(v-c_y)\,D(u,v)}{f_y},\quad Z = D(u,v)$$

For TUM fr1/desk scaled to 960×540: $f_x=775.95,\ f_y=581.06,\ c_x=477.9,\ c_y=287.2$.

### 2 · UniDepth / Depth Anything V2 — Neural Depth Estimation

Both models use a **Dense Prediction Transformer (DPT)** decoder on a **ViT-Small** backbone.

**Patch embedding:** an image $I\in\mathbb{R}^{H\times W\times 3}$ is split into $N=\frac{HW}{p^2}$ non-overlapping $p\times p$ patches ($p=14$ for ViT-Small), linearly projected to token dimension $d=384$:

$$\mathbf{z}_0 = [\mathbf{x}_{\text{cls}};\, \mathbf{x}_1\mathbf{E};\ldots;\mathbf{x}_N\mathbf{E}] + \mathbf{E}_{\text{pos}}, \qquad \mathbf{E}\in\mathbb{R}^{(p^2\cdot 3)\times d}$$

**Transformer layers** ($L=12$) apply multi-head self-attention + feed-forward:

$$\text{MSA}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

**DPT decoder** reassembles tokens at 4 different scales into a dense feature map, which is upsampled and passed through a convolutional head to produce:

$$D(u,v)\in\mathbb{R}^+ \quad[\text{metres}]$$

The **metric** variant trains an additional scale+shift calibration head to produce absolute depth without any post-hoc normalisation.

**Training loss** (scale-invariant + gradient + normal consistency):

$$\mathcal{L} = \lambda_\text{si}\,\mathcal{L}_\text{si} + \lambda_\nabla\,\mathcal{L}_\nabla + \lambda_n\,\mathcal{L}_n$$

where $\mathcal{L}_\text{si} = \frac{1}{N}\sum_i d_i^2 - \frac{1}{N^2}\!\left(\sum_i d_i\right)^{\!2}$,  $d_i = \log \hat{D}_i - \log D_i^*$.

### 3 · ORB Feature Extraction

For each pyramid level $s$ (scale factor 1.2, 8 levels), FAST corners are detected and retained by Harris score. Orientation is computed via the **intensity centroid**:

$$m_{pq}=\sum_{x,y} x^p y^q I(x,y),\qquad \theta = \text{atan2}(m_{01}, m_{10})$$

The **steered BRIEF** descriptor samples 256 random pixel pairs around the keypoint, rotated by $\theta$, producing a 256-bit binary vector. Descriptor matching uses **Hamming distance** with ratio test ($\tau=0.7$).

### 4 · RGB-D Tracking via PnP + RANSAC

Given $n$ matched features $\{(u_i,v_i)\}$ with associated depths $\{D_i\}$, back-project to 3-D:

$$\mathbf{P}_i = \left[\frac{(u_i-c_x)D_i}{f_x},\ \frac{(v_i-c_y)D_i}{f_y},\ D_i\right]^\top$$

Estimate the camera pose $\mathbf{T}\in SE(3)$ by minimising reprojection error:

$$\mathbf{T}^* = \argmin_{\mathbf{T}} \sum_{i=1}^n \left\|\pi_\mathbf{K}(\mathbf{T}\cdot\mathbf{P}_i) - \begin{pmatrix}u_i\\v_i\end{pmatrix}\right\|^2$$

Solved with **iterative PnP + RANSAC** (inlier threshold 8 px).

### 5 · Local Bundle Adjustment

Over a co-visible keyframe window, jointly optimise poses $\{\mathbf{T}_j\}$ and map-point positions $\{\mathbf{P}_k\}$:

$$\min_{\{\mathbf{T}_j\},\{\mathbf{P}_k\}} \sum_{j,k} \rho\!\left(\left\|\mathbf{p}_{jk} - \pi_\mathbf{K}(\mathbf{T}_j\mathbf{P}_k)\right\|^2_{\Sigma_{jk}}\right)$$

where $\rho$ is the Huber robust kernel and $\Sigma_{jk}$ is the observation covariance. Solved with **g2o** (Gauss-Newton / Levenberg–Marquardt).

### 6 · Loop Closure via DBoW2

Each keyframe is represented as a **Bag-of-Words** vector $\mathbf{v}\in\mathbb{R}^V$ over a visual vocabulary of $V=10^6$ ORB words. A candidate loop is accepted when:

$$s(\mathbf{v}_\text{query}, \mathbf{v}_\text{cand}) > \tau_\text{loop}$$

Geometric verification computes a **Sim3** alignment (SE3 for RGB-D, since scale is fixed), which is then propagated through the **Essential Graph** to correct accumulated drift.

### 7 · Pose Representation

ORB-SLAM2 outputs $\mathbf{T}_{cw}\in SE(3)$ (world → camera). We invert to get the camera position in world frame:

$$\mathbf{T}_{wc} = \mathbf{T}_{cw}^{-1} = \begin{pmatrix}\mathbf{R}_{wc} & \mathbf{t}_{wc}\\0 & 1\end{pmatrix}$$

$\mathbf{R}_{wc}$ is converted to quaternion $\mathbf{q}=(x,y,z,w)$ via the **Shepperd method** and published as `geometry_msgs/PoseStamped`.

---

## Hardware & Software Requirements

| Component | Specification |
|---|---|
| **Board** | NVIDIA Jetson Orin Nano Super Developer Kit (8 GB) |
| **OS** | Ubuntu 22.04 · JetPack 6.x (L4T R36.4+) |
| **Camera** | IMX219 CSI (Raspberry Pi Camera v2 compatible) |
| **ROS** | ROS 2 Humble |
| **CUDA** | 12.x (via JetPack) |
| **cuDNN** | 9.3 |
| **Python** | 3.10 |
| **Storage** | ≥ 32 GB (model + dataset) |

---

## Installation — From Scratch

### 1 · System Dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
  build-essential cmake git wget curl \
  libboost-all-dev libeigen3-dev \
  libglew-dev libpangolin-dev \
  python3-pip python3-dev
```

### 2 · ROS 2 Humble

```bash
sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) \
  signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu jammy main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list
sudo apt update
sudo apt install -y ros-humble-desktop \
  python3-colcon-common-extensions \
  ros-humble-cv-bridge ros-humble-message-filters
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 3 · PyTorch for JetPack 6 (CUDA 12 + cuDNN 9)

```bash
pip3 install --upgrade pip

# Official NVIDIA wheel for JetPack 6 / L4T R36
wget https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/\
torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

pip3 install torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
pip3 install "numpy<2"    # required by ROS cv_bridge

# Verify GPU is available
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### 4 · Depth Model

```bash
# For UniDepth (used in dataset mode)
pip3 install unidepth-pytorch

# For Depth Anything V2 (used in live camera mode)
pip3 install transformers accelerate

# Pre-download weights (optional, avoids first-run delay)
python3 -c "
from transformers import pipeline
pipeline('depth-estimation',
    model='depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf')
print('Model cached.')
"
```

### 5 · ORB-SLAM2 + Python Bindings

```bash
# Build ORB-SLAM2
git clone https://github.com/raulmur/ORB_SLAM2.git ~/ORB_SLAM2
cd ~/ORB_SLAM2 && chmod +x build.sh && ./build.sh
cd build && sudo make install      # installs to /usr/local

# Extract vocabulary
cd ~/ORB_SLAM2/Vocabulary && tar -xf ORBvoc.txt.tar.gz

# Build Python bindings (jskinn fork)
git clone https://github.com/jskinn/ORB_SLAM2-PythonBindings.git \
    ~/ORB_SLAM2-PythonBindings
cd ~/ORB_SLAM2-PythonBindings

# Apply fixes for Python 3.10 + C++14
sed -i 's/c++11/c++14/g; s/c++0x/c++14/g'                        CMakeLists.txt
sed -i 's/PythonLibs 3\.5/PythonLibs 3.10/g'                     CMakeLists.txt
sed -i 's/python-py35/python310/g'                                CMakeLists.txt
sed -i 's|python3\.5/dist-packages|python3/dist-packages|g'      CMakeLists.txt
sed -i 's/return NUMPY_IMPORT_ARRAY_RETVAL;/import_array(); return NULL;/' \
    src/ORBSlamPython.cpp
sed -i 's/set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall/set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive -Wall/' \
    CMakeLists.txt

mkdir build && cd build
cmake ~/ORB_SLAM2-PythonBindings \
    -DORB_SLAM2_DIR=/usr/local \
    -DCMAKE_CXX_STANDARD=14
make -j$(nproc) && sudo make install

python3 -c "import orbslam2; print('orbslam2 bindings OK')"
```

### 6 · Clone & Build This Package

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone https://github.com/YOUR_USERNAME/pseudo_rgbd_slam.git

cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select pseudo_rgbd_slam --symlink-install
source install/setup.bash
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

---

## Running — TUM fr1/desk Dataset

### Download Dataset

```bash
mkdir -p ~/datasets/tum && cd ~/datasets/tum
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/\
rgbd_dataset_freiburg1_desk.tgz
tar -xzf rgbd_dataset_freiburg1_desk.tgz
```

### Option A — Single Launch File

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 launch pseudo_rgbd_slam slam_pipeline.launch.py \
    dataset_path:=$HOME/datasets/tum/rgbd_dataset_freiburg1_desk \
    vocab_path:=$HOME/ORB_SLAM2/Vocabulary/ORBvoc.txt \
    settings_path:=$HOME/ros2_ws/src/pseudo_rgbd_slam/config/orb_slam2_rgbd.yaml
```

### Option B — Four Separate Terminals

```bash
# Terminal 1 — Node A: RGB Broadcaster
# fps=3 is important — depth inference takes ~300 ms, so publish slowly
ros2 run pseudo_rgbd_slam node_a \
    --ros-args \
    -p dataset_path:=$HOME/datasets/tum/rgbd_dataset_freiburg1_desk \
    -p fps:=3.0
```

```bash
# Terminal 2 — Node B: Neural Depth Estimator
ros2 run pseudo_rgbd_slam node_b --ros-args -p device:=auto
```

```bash
# Terminal 3 — Node C: ORB-SLAM2 RGB-D
ros2 run pseudo_rgbd_slam node_c \
    --ros-args \
    -p vocab_path:=$HOME/ORB_SLAM2/Vocabulary/ORBvoc.txt \
    -p settings_path:=$HOME/ros2_ws/src/pseudo_rgbd_slam/config/orb_slam2_rgbd.yaml
```

```bash
# Terminal 4 — Node D: Visualizer (RGB + Depth + Trajectory)
ros2 run pseudo_rgbd_slam visualizer
```

```bash
# Terminal 5 — Verify SLAM is producing poses
ros2 topic hz /slam/pose
```

---

## Running — Live Camera (Jetson Orin Nano Super + IMX219)

> **Prerequisites:** `nvargus-daemon` must be running.
> ```bash
> sudo systemctl status nvargus-daemon   # check
> sudo systemctl restart nvargus-daemon  # if not running
> ```

```bash
# Terminal 1 — Node A: IMX219 via GStreamer
ros2 run pseudo_rgbd_slam node_a \
    --ros-args \
    -p source:=camera \
    -p sensor_id:=0 \
    -p publish_rate:=5.0
```

```bash
# Terminal 2 — Node B: Depth Anything V2 on GPU
ros2 run pseudo_rgbd_slam node_b \
    --ros-args \
    -p model_name:=depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf \
    -p device:=auto \
    -p input_size:=392
```

```bash
# Terminal 3 — Node C: ORB-SLAM2 RGB-D
ros2 run pseudo_rgbd_slam node_c \
    --ros-args \
    -p vocab_path:=$HOME/ORB_SLAM2/Vocabulary/ORBvoc.txt \
    -p settings_path:=$HOME/ros2_ws/src/pseudo_rgbd_slam/config/orb_slam2_rgbd.yaml
```

```bash
# Terminal 4 — Node D: Visualizer
ros2 run pseudo_rgbd_slam visualizer
```

---

## RViz2 Visualisation

```bash
rviz2 -d ~/ros2_ws/src/pseudo_rgbd_slam/config/slam_view.rviz
```

Add the following displays manually if using default config:

| Display | Topic | Type |
|---|---|---|
| Image (RGB) | `/camera/image_raw` | Image |
| Image (Depth) | `/depth/image_visual` | Image |
| Path | `/slam/trajectory` | Path |
| PointCloud2 | `/slam/map_points` | PointCloud2 |
| Pose | `/slam/pose` | Pose |

Set **Fixed Frame** to `map`.

---

## Performance on Jetson Orin Nano Super (8 GB)

| Stage | Mode | Latency | Rate |
|---|---|---|---|
| Node A: Camera capture | GStreamer HW (NVMM) | ~0 ms | 5–15 Hz |
| Node B: Depth inference (GPU FP16) | CUDA, 392px input | ~90–150 ms | 6–10 Hz |
| Node B: Depth inference (CPU FP32) | Fallback | ~100 s ⚠ | 0.01 Hz |
| Node C: ORB extraction + PnP | CPU | ~25–50 ms | 6–10 Hz |
| Node C: Local BA | CPU (g2o) | ~10–30 ms | — |
| **End-to-end latency** | GPU path | **~150–250 ms** | **5–8 Hz** |

> GPU depth inference is **~1000×** faster than CPU. Always run with CUDA enabled.

---

## Project Structure

```
pseudo_rgbd_slam/
├── pseudo_rgbd_slam/
│   ├── node_a_broadcaster.py     # Node A: RGB source (camera / dataset)
│   ├── node_b_depth_estimator.py # Node B: UniDepth / Depth Anything V2
│   ├── node_c_slam.py            # Node C: ORB-SLAM2 RGB-D
│   ├── node_d_visualizer.py      # Node D: OpenCV combined display
│   └── utils/
├── config/
│   └── orb_slam2_rgbd.yaml       # Camera intrinsics + ORB parameters
├── launch/
│   └── slam_pipeline.launch.py   # Single-command launch (all nodes)
├── docs/
│   └── images/                   # Screenshots for this README
└── README.md
```

---

## Troubleshooting

**`libcudnn.so.8: cannot open shared object file`**
You have cuDNN 9 but an old PyTorch wheel. Use the JetPack 6 wheel from Step 3 above.

**SLAM status stays `NOT_INITIALIZED`**
Move the camera slowly across a textured surface. The system needs ≥ 20 feature matches across 2 frames to initialise. Blank walls or ceilings will not initialise.

**Depth inference very slow (~100 s/frame)**
CUDA is not active. Check `torch.cuda.is_available()` and verify `CUDA_VISIBLE_DEVICES` is not set to `""`.

**`cv_bridge` import fails**
Source ROS before running: `source /opt/ros/humble/setup.bash`

**`numpy` version conflict**
`pip3 install "numpy<2"` — the system cv_bridge was compiled against NumPy 1.x.

**No map points in RViz**
In the PointCloud2 display set **Durability** to `Transient Local` to receive messages published before RViz connected.

---

## Acknowledgements

- [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2) — Mur-Artal et al., IEEE TRO 2017
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) — Yang et al., NeurIPS 2024
- [UniDepth](https://github.com/lpiccinelli-eth/UniDepth) — Piccinelli et al., CVPR 2024
- [TUM RGB-D Benchmark](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) — Sturm et al., IROS 2012

---

## License

MIT © 2025
