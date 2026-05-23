# Building-a-Pseudo-RGB-D-SLAM-ROS-Package
\documentclass[10pt,twocolumn]{article}

% ── Packages ──────────────────────────────────────────────────────────────────
\usepackage[margin=1.8cm,top=2cm,bottom=2cm]{geometry}
\usepackage{amsmath,amssymb,bm}
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{enumitem}
\usepackage{microtype}
\usepackage{titlesec}
\usepackage{parskip}
\usepackage{float}
\usepackage{tabularx}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage{mathtools}
\usepackage{algorithm}
\usepackage{algpseudocode}

% ── Colours ───────────────────────────────────────────────────────────────────
\definecolor{nvidiagreen}{RGB}{118,185,0}
\definecolor{darkgray}{RGB}{50,50,50}
\definecolor{lightgray}{RGB}{240,240,240}
\definecolor{accentblue}{RGB}{0,102,204}

% ── Section formatting ────────────────────────────────────────────────────────
\titleformat{\section}{\large\bfseries\color{darkgray}}{{\color{accentblue}\thesection.}}{0.5em}{}[\vspace{-4pt}\color{accentblue}\rule{\linewidth}{0.8pt}]
\titleformat{\subsection}{\normalsize\bfseries\color{darkgray}}{\thesubsection}{0.5em}{}
\titlespacing{\section}{0pt}{10pt}{4pt}
\titlespacing{\subsection}{0pt}{6pt}{2pt}

% ── Hyperref ──────────────────────────────────────────────────────────────────
\hypersetup{colorlinks,linkcolor=accentblue,citecolor=accentblue,urlcolor=accentblue}

% ── Operators ─────────────────────────────────────────────────────────────────
\DeclareMathOperator*{\argmin}{arg\,min}
\newcommand{\T}{^\mathsf{T}}

% ═════════════════════════════════════════════════════════════════════════════
\begin{document}

% ── Title ─────────────────────────────────────────────────────────────────────
\twocolumn[{%
\begin{center}
  {\LARGE\bfseries Pseudo RGB-D SLAM via Neural Depth Estimation}\\[6pt]
  {\large A Modular ROS~2 Pipeline on Jetson Orin Nano Super}\\[8pt]
  {\normalsize\textit{IMX219 Camera $\cdot$ Depth Anything V2 $\cdot$ ORB-SLAM2 RGB-D}}\\[10pt]
  \hrule height 1pt
  \vspace{10pt}
\end{center}
}]

% ── Abstract ──────────────────────────────────────────────────────────────────
\begin{abstract}
\noindent
We present a modular ROS~2 pipeline that replaces a physical depth sensor with a monocular neural depth estimator, enabling RGB-D SLAM on any single-camera system.
Three independent nodes communicate via standard ROS topics:
\textbf{Node~A} acquires RGB frames from an IMX219 CSI camera (via GStreamer NVMM) or the TUM~fr1/desk dataset;
\textbf{Node~B} runs \emph{Depth Anything V2}~\cite{depthanything2} (ViT-Small, metric indoor variant) at FP16 on the Jetson GPU, producing per-frame metric depth at 6--10~Hz;
\textbf{Node~C} feeds the fused RGB+depth stream into \emph{ORB-SLAM2}~\cite{orbslam2} (RGB-D mode) for real-time tracking, local bundle adjustment, and loop closure.
The complete pipeline achieves 5--8~Hz end-to-end on a 2~GB GPU budget, with a latency of 150--250~ms.
\end{abstract}

% ─────────────────────────────────────────────────────────────────────────────
\section{Introduction}

Simultaneous Localisation and Mapping (SLAM) using RGB-D cameras requires per-pixel metric depth, traditionally obtained from structured-light or time-of-flight sensors.
Recent advances in monocular metric depth estimation~\cite{depthanything2,unidepth} have reached accuracy levels that make them viable substitutes, enabling SLAM on any RGB-only platform.

This work builds a \emph{pseudo} RGB-D SLAM pipeline:
(i)~a live IMX219 CSI camera provides colour frames at up to 15~Hz;
(ii)~a transformer-based depth network predicts metric depth at each frame;
(iii)~ORB-SLAM2 RGB-D mode uses the fused data for pose estimation, mapping, and loop closure.
The system runs on a Jetson Orin Nano Super Developer Kit with a 2~GB effective GPU budget, operating at 5--8~Hz end-to-end.

% ─────────────────────────────────────────────────────────────────────────────
\section{System Architecture}

The pipeline is split into four ROS~2 nodes (Fig.~\ref{fig:architecture}), each running as an independent process.

\subsection{Node A — RGB Broadcaster}
Captures 960$\times$540 BGR frames from the IMX219 via the GStreamer pipeline:
\begin{center}
\texttt{nvarguscamerasrc} $\to$ \texttt{nvvidconv} $\to$ \texttt{appsink}
\end{center}
All processing uses NVIDIA NVMM zero-copy memory.
For validation, Node~A can replay the TUM~fr1/desk dataset~\cite{tum} at a configurable rate.
Publishes \texttt{/camera/image\_raw} and \texttt{/camera/camera\_info}.

\subsection{Node B — Neural Depth Estimator}
Subscribes to the RGB stream, runs inference, and publishes a \texttt{float32} depth image (metres) to \texttt{/depth/image\_raw}.
A dedicated background thread performs inference while the ROS callback returns immediately, so the executor never blocks.
The latest frame is always processed; stale frames are dropped.
A plasma-colourmap visualisation is published to \texttt{/depth/image\_visual} for display.

\subsection{Node C — Pseudo SLAM}
Subscribes to both RGB and depth. A manual cache-based synchroniser pairs the most recent depth with the latest RGB and submits the pair to ORB-SLAM2.
Publishes pose (\texttt{PoseStamped}), trajectory (\texttt{Path}), map points (\texttt{PointCloud2}), and tracking status.

\subsection{Node D — Visualizer}
A dedicated OpenCV window shows all three key streams side-by-side: RGB feed, depth colourmap, and 2-D trajectory.

% ─────────────────────────────────────────────────────────────────────────────
\section{Depth Estimation Model}

\subsection{Architecture}
\emph{Depth Anything V2}~\cite{depthanything2} uses a \textbf{Dense Prediction Transformer (DPT)}~\cite{dpt} decoder on a \textbf{ViT-Small} backbone~\cite{vit}.

\paragraph{Patch embedding.}
The input image $\mathbf{I}\in\mathbb{R}^{H\times W\times 3}$ is divided into $N=HW/p^2$ patches of size $p{=}14$, linearly projected to tokens of dimension $d{=}384$:
\begin{equation}
  \mathbf{z}_0 = [\mathbf{x}_\text{cls};\,\mathbf{x}_1\mathbf{E};\ldots;\mathbf{x}_N\mathbf{E}] + \mathbf{E}_\text{pos}
\end{equation}

\paragraph{Multi-head self-attention.}
$L{=}12$ transformer layers apply:
\begin{equation}
  \text{MSA}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \text{softmax}\!\left(\!\frac{\mathbf{Q}\mathbf{K}\T}{\sqrt{d_k}}\!\right)\!\mathbf{V}
\end{equation}

\paragraph{DPT decoder.}
Tokens from layers $\{3,6,9,12\}$ are reassembled into feature maps at $\{1/32,1/16,1/8,1/4\}$ resolution, fused by a convolutional head, and upsampled to the full resolution to produce metric depth $D(u,v)\in\mathbb{R}^+$ in metres.

\subsection{Metric Calibration}
The \emph{Indoor-Small} variant appends a scale-and-shift head trained on mixed indoor depth datasets.
No post-hoc normalisation is needed: the output is absolute depth in metres.

\subsection{Training Loss}
\begin{equation}
  \mathcal{L} = \lambda_\text{si}\,\mathcal{L}_\text{si} + \lambda_\nabla\,\mathcal{L}_\nabla + \lambda_n\,\mathcal{L}_n
\end{equation}
where the scale-invariant term is:
\begin{equation}
  \mathcal{L}_\text{si} = \frac{1}{N}\sum_i d_i^2 - \frac{\alpha}{N^2}\!\left(\sum_i d_i\right)^{\!2},\quad d_i=\log\hat{D}_i - \log D_i^*
\end{equation}
$\mathcal{L}_\nabla$ is the multi-scale image-gradient loss and $\mathcal{L}_n$ is the surface-normal consistency loss.

\subsection{Runtime Optimisation on Jetson}
FP16 inference on the Jetson GPU (Ampere, effective 2~GB) reduces memory bandwidth and compute by $\sim\!2\times$ compared to FP32.
The input is resized to $392\times392$ (a multiple of $p{=}14$) before the patch embedding to further reduce the token count, achieving 6--10~Hz.

% ─────────────────────────────────────────────────────────────────────────────
\section{SLAM — ORB-SLAM2 RGB-D}

\subsection{Camera Model \& Back-Projection}
Under the pinhole model with intrinsic matrix $\mathbf{K}$:
\begin{equation}
  \begin{pmatrix}u\\v\\1\end{pmatrix} = \frac{1}{Z}\mathbf{K}\mathbf{P},\qquad \mathbf{K}=\begin{pmatrix}f_x&0&c_x\\0&f_y&c_y\\0&0&1\end{pmatrix}
\end{equation}
Back-projection from the predicted depth map $D(u,v)$:
\begin{equation}
  \mathbf{P} = \begin{bmatrix}\dfrac{(u-c_x)D(u,v)}{f_x} & \dfrac{(v-c_y)D(u,v)}{f_y} & D(u,v)\end{bmatrix}\T
  \label{eq:backproj}
\end{equation}
For TUM fr1/desk scaled to 960$\times$540: $f_x{=}775.95$, $f_y{=}581.06$, $c_x{=}477.9$, $c_y{=}287.2$.

\subsection{ORB Feature Extraction}
An 8-level image pyramid (scale factor 1.2) is built.
At each level, \textbf{FAST} corners are detected and filtered by Harris score; 1000 features are retained per frame.
Orientation is computed via the intensity centroid:
\begin{equation}
  \theta = \text{atan2}(m_{01},\,m_{10}),\quad m_{pq}=\textstyle\sum_{x,y}x^py^qI(x,y)
\end{equation}
The \textbf{steered BRIEF} descriptor applies 256 random binary tests rotated by $\theta$, yielding a 256-bit vector matched by Hamming distance with ratio test $\tau{=}0.7$.

\subsection{RGB-D Tracking via PnP}
Matched 2-D features $\{(u_i,v_i)\}$ are back-projected to 3-D via Eq.~\eqref{eq:backproj}.
Camera pose $\mathbf{T}\in SE(3)$ is recovered by minimising reprojection error:
\begin{equation}
  \mathbf{T}^* = \argmin_{\mathbf{T}} \sum_{i=1}^n \left\|\pi_\mathbf{K}(\mathbf{T}\cdot\mathbf{P}_i) - \begin{pmatrix}u_i\\v_i\end{pmatrix}\right\|^2
\end{equation}
Solved with iterative PnP and \textbf{RANSAC} (inlier threshold 8~px).

\subsection{Local Bundle Adjustment}
Over a co-visible keyframe window, poses $\{\mathbf{T}_j\}$ and map-point positions $\{\mathbf{P}_k\}$ are jointly optimised:
\begin{equation}
  \min_{\{\mathbf{T}_j\},\{\mathbf{P}_k\}} \sum_{j,k} \rho\!\left(\left\|\mathbf{p}_{jk} - \pi_\mathbf{K}(\mathbf{T}_j\mathbf{P}_k)\right\|^2_{\Sigma_{jk}}\right)
\end{equation}
where $\rho(\cdot)$ is the Huber kernel.
Solved with \textbf{g2o} (Levenberg--Marquardt).

\subsection{Loop Closure}
Each keyframe is encoded as a \textbf{DBoW2} bag-of-words vector $\mathbf{v}\in\mathbb{R}^V$.
A loop candidate is accepted when $s(\mathbf{v}_q,\mathbf{v}_c)>\tau_\text{loop}$.
A \textbf{Sim3} constraint (SE(3) for RGB-D) aligns the two views and drift is corrected via Essential Graph optimisation.

\subsection{Pose Representation}
ORB-SLAM2 outputs $\mathbf{T}_{cw}\in SE(3)$ (world$\to$camera).
The camera position in world frame is:
\begin{equation}
  \mathbf{T}_{wc} = \mathbf{T}_{cw}^{-1} = \begin{pmatrix}\mathbf{R}_{wc}&\mathbf{t}_{wc}\\0&1\end{pmatrix}
\end{equation}
$\mathbf{R}_{wc}$ is converted to unit quaternion $\mathbf{q}=(x,y,z,w)$ via the Shepperd method and published as \texttt{geometry\_msgs/PoseStamped}.

% ─────────────────────────────────────────────────────────────────────────────
\section{Results}

\subsection{Live Camera Demo}

Fig.~\ref{fig:rviz_live} shows RViz2 running the live pipeline on the Jetson Orin Nano Super.
The depth map (Plasma colourmap) and the camera trajectory (green path) update in real time.
Fig.~\ref{fig:visualizer} shows the combined Node~D visualizer window with RGB, depth, and trajectory simultaneously.

\begin{figure}[H]
  \centering
  \begin{subfigure}[b]{0.48\linewidth}
    \includegraphics[width=\linewidth]{docs/images/rviz_depth_trajectory.png}
    \caption{Depth map + trajectory}
    \label{fig:rviz_depth}
  \end{subfigure}\hfill
  \begin{subfigure}[b]{0.48\linewidth}
    \includegraphics[width=\linewidth]{docs/images/rviz_map_points.png}
    \caption{3-D map points + pose}
    \label{fig:rviz_map}
  \end{subfigure}
  \caption{RViz2 output: live IMX219 camera on Jetson Orin Nano Super.}
  \label{fig:rviz_live}
\end{figure}

\begin{figure}[H]
  \centering
  \includegraphics[width=\linewidth]{docs/images/visualizer_combined.png}
  \caption{Node D visualizer: RGB feed (left), depth colourmap (centre), 2-D trajectory (right).}
  \label{fig:visualizer}
\end{figure}

\subsection{Performance}

\begin{table}[H]
  \centering
  \small
  \caption{Per-stage latency on Jetson Orin Nano Super (8~GB, 2~GB GPU budget).}
  \label{tab:perf}
  \begin{tabularx}{\linewidth}{@{}Xcc@{}}
    \toprule
    \textbf{Stage} & \textbf{Latency} & \textbf{Rate} \\
    \midrule
    Node A: GStreamer capture  & $\sim$0~ms   & 5--15~Hz \\
    Node B: Depth (GPU FP16)   & 90--150~ms   & 6--10~Hz \\
    Node B: Depth (CPU, fallback) & $\sim$100~s {\small\color{red}$\times$} & 0.01~Hz \\
    Node C: ORB + PnP          & 25--50~ms    & 6--10~Hz \\
    Node C: Local BA (g2o)     & 10--30~ms    & —        \\
    \midrule
    \textbf{End-to-end (GPU)}  & \textbf{150--250~ms} & \textbf{5--8~Hz} \\
    \bottomrule
  \end{tabularx}
\end{table}

\subsection{Depth Quality vs Physical Sensor}

\begin{table}[H]
  \centering
  \small
  \caption{Comparison: neural depth vs RealSense D435.}
  \label{tab:depth_cmp}
  \begin{tabularx}{\linewidth}{@{}Xcc@{}}
    \toprule
    & \textbf{D435} & \textbf{Depth Anything V2} \\
    \midrule
    Depth range     & 0.1--10~m & 0--20~m \\
    Indoor RMSE     & $\sim$2~mm & 50--150~mm \\
    Scale drift     & None & None (metric) \\
    Transparent obj.& Fails & Partially fails \\
    Extra hardware  & USB sensor & None \\
    Latency         & $<$1~ms & 90--150~ms \\
    \bottomrule
  \end{tabularx}
\end{table}

% ─────────────────────────────────────────────────────────────────────────────
\section{Key Design Decisions}

\paragraph{Why Depth Anything V2 over UniDepth?}
The HuggingFace \texttt{pipeline} API makes integration trivial; the Indoor-Small variant outputs absolute metres without post-processing; and the ViT-Small backbone fits comfortably in 2~GB GPU memory at FP16.

\paragraph{Why manual cache sync instead of \texttt{ApproximateTimeSynchronizer}?}
Node B preserves the original RGB timestamp on its depth output, so any synchroniser would work.
However, our manual cache sync (pair latest depth to latest RGB) has \emph{zero queue delay} and is immune to the parameter-tuning pitfalls of ATS.

\paragraph{Why 5--8~Hz instead of 30~Hz?}
The bottleneck is depth inference ($\sim$100~ms on a 2~GB GPU).
With TensorRT export of the ViT backbone the expected throughput rises to 12--18~Hz.
A physical depth sensor removes the bottleneck entirely.

\paragraph{Depth quality for SLAM.}
At 50--150~mm RMSE, the predicted depth is noisy enough to affect BA precision but rarely degrades tracking entirely.
The main failure mode is large featureless surfaces (blank walls, ceilings).

% ─────────────────────────────────────────────────────────────────────────────
\section{Conclusion}

We demonstrated that a monocular neural depth estimator can successfully replace a physical RGB-D sensor in an ORB-SLAM2 pipeline, running at 5--8~Hz on a 2~GB GPU embedded platform.
The system achieves real-time visual odometry, sparse 3-D mapping, and loop closure on both the TUM fr1/desk benchmark and live indoor scenes captured by an IMX219 camera on a Jetson Orin Nano Super.

Future work includes TensorRT optimisation of the depth model ($\sim$2$\times$ speedup), temporal depth consistency filtering to reduce per-frame noise, and integration with a dense reconstruction backend.

% ─────────────────────────────────────────────────────────────────────────────
\begin{thebibliography}{9}
\bibitem{orbslam2}
R.~Mur-Artal and J.~D.~Tard\'os, ``ORB-SLAM2: An Open-Source SLAM System for Monocular, Stereo, and RGB-D Cameras,'' \textit{IEEE Trans.\ Robotics}, 2017.

\bibitem{depthanything2}
L.~Yang et~al., ``Depth Anything V2,'' \textit{NeurIPS}, 2024.

\bibitem{unidepth}
L.~Piccinelli et~al., ``UniDepth: Universal Monocular Metric Depth Estimation,'' \textit{CVPR}, 2024.

\bibitem{dpt}
R.~Ranftl et~al., ``Vision Transformers for Dense Prediction,'' \textit{ICCV}, 2021.

\bibitem{vit}
A.~Dosovitskiy et~al., ``An Image is Worth 16x16 Words,'' \textit{ICLR}, 2021.

\bibitem{tum}
J.~Sturm et~al., ``A Benchmark for the Evaluation of RGB-D SLAM Systems,'' \textit{IROS}, 2012.

\bibitem{dbow2}
D.~G\'alvez-L\'opez and J.~D.~Tard\'os, ``Bags of Binary Words for Fast Place Recognition,'' \textit{IEEE Trans.\ Robotics}, 2012.
\end{thebibliography}

\end{document}
