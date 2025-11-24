FROM --platform=linux/amd64 ubuntu:16.04

# 基础工具
RUN apt-get update && apt-get install -y \
    git wget unzip curl build-essential cmake \
    libgl1-mesa-dev libglfw3 libglfw3-dev \
    libglew-dev patchelf python3.5 python3.5-dev python3.5-venv \
    python3-tk

# 默认 python 命令使用 python3.5
RUN ln -sf /usr/bin/python3.5 /usr/bin/python

# 安装 pip（不要安装 python3-pip！）
RUN wget https://bootstrap.pypa.io/pip/3.5/get-pip.py && \
    python get-pip.py && rm get-pip.py

# 降级 pip（pip 10 才能正常支持 Python3.5）
RUN python -m pip install --upgrade "pip<10" setuptools

# 安装 TensorFlow 0.12.1（最后兼容 Python3.5 的版本）
# 先安装兼容 Python 3.5 的 certifi 和 requests（避免 os.PathLike 错误）
RUN pip install 'certifi<2021.0.0' 'requests<3.0.0' \
    numpy==1.11.3 scipy==0.18.1

RUN wget https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp35-cp35m-linux_x86_64.whl -O /tmp/tensorflow-0.12.1-cp35-cp35m-linux_x86_64.whl

RUN pip install /tmp/tensorflow-0.12.1-cp35-cp35m-linux_x86_64.whl \
    'protobuf<3.0.0'

# 兼容 mujoco-py 所需依赖
# 注意：不安装 python3-opengl，使用 pip 安装的 PyOpenGL 以避免版本冲突
RUN apt-get update && apt-get install -y \
    libosmesa6-dev libgl1-mesa-glx

# 安装 SWIG（pybox2d 编译需要）
RUN apt-get update && apt-get install -y swig

# 安装其他依赖
# 注意：certifi 需要降级到兼容 Python 3.5 的版本（os.PathLike 是 Python 3.6+ 的特性）
RUN pip install 'certifi<2021.0.0' \
    'requests<3.0.0' \
    matplotlib==2.0.2 \
    plotly==1.9.10 \
    gym==0.8.0 \
    pygame==1.9.6 \
    path.py \
    joblib==0.9.4 \
    python-dateutil \
    mako \
    h5py \
    scikit-learn \
    Pillow \
    pyprind \
    ipdb \
    'PyOpenGL==3.1.0' \
    nose2 \
    pyzmq \
    msgpack-python \
    cached_property \
    Cython \
    progressbar2 \
    git+https://github.com/pybox2d/pybox2d.git@2.3.2#egg=pybox2d

# 安装 PyTorch（最后一代支持 Python 3.5 的 CPU 版本）
# 注意：需要联网从 PyTorch 官方源下载 wheel
RUN pip install --no-cache-dir \
    torch==1.4.0 \
    torchvision==0.5.0 \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

# 配置 matplotlib 使用非 GUI 后端（适合服务器环境）
ENV MPLBACKEND=Agg

# 安装 Theano 和 Lasagne（从 git 安装，兼容 Python 3.5）
RUN pip install git+https://github.com/Theano/Theano.git@adfe319ce6b781083d8dc3200fb4481b00853791#egg=Theano \
    git+https://github.com/neocxi/Lasagne.git@484866cf8b38d878e92d521be445968531646bb8#egg=Lasagne

# 安装 mujoco_py（MuJoCo 物理引擎的 Python 绑定）
# 注意：mujoco_py 需要 MuJoCo 二进制文件
# 使用旧版本以兼容 Python 3.5 和 Pillow 7.2.0
# MuJoCo 文件应该在运行时通过卷挂载，或者复制到镜像中
# 设置 MuJoCo 环境变量（如果 MuJoCo 已挂载）
ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mjpro131
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mjpro131/bin
# 预安装兼容的依赖（旧版本的 imageio 和 Pillow）
# 注意：mujoco_py 需要在挂载 MuJoCo 文件后安装
RUN pip install 'imageio<2.1.0' || echo "Warning: Pre-installing mujoco_py dependencies. Install mujoco_py<0.5.8 after mounting MuJoCo files."

# 工作目录
WORKDIR /workspace

# 设置 PYTHONPATH 以包含 rllab 模块
# 这样可以直接导入 rllab 而不需要安装
ENV PYTHONPATH=/workspace:$PYTHONPATH

# 复制并安装本地的 gym-adv（包含 adversarial 环境注册）
# 注意：Docker COPY 不支持条件复制，所以需要根据构建方式选择：
#
# 方式 1：从父目录构建（推荐，包含 gym-adv）
#   cd /Users/baishuwen/Desktop/mlrl/robust-adversarial-rl/src
#   docker build -f rllab-adv/Dockerfile -t rllab-adv .
#   然后取消下面的注释：
# COPY gym-adv /tmp/gym-adv
# RUN cd /tmp/gym-adv && pip install -e .
#
# 方式 2：从 rllab-adv 目录构建（不包含 gym-adv，需要在运行时安装）
#   cd /Users/baishuwen/Desktop/mlrl/robust-adversarial-rl/src/rllab-adv
#   docker build -t rllab-adv .
#   然后在运行时挂载并安装：
#   docker run -v /Users/baishuwen/Desktop/mlrl/robust-adversarial-rl/src/gym-adv:/tmp/gym-adv -it rllab-adv bash
#   然后在容器内运行: pip install -e /tmp/gym-adv
#
# 当前配置：不包含 gym-adv（从 rllab-adv 目录构建时使用）
# 如需包含，请从父目录构建并取消下面的注释
# COPY gym-adv /tmp/gym-adv
# RUN cd /tmp/gym-adv && pip install -e .

CMD ["bash"]