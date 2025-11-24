
docker build -t rllab-adv .

docker run -it --rm \
  -v /Users/baishuwen/Desktop/mlrl_project/src:/workspace \
  -v /Users/baishuwen/Desktop/mlrl_project/src/gym-adv:/tmp/gym-adv \
  -v /Users/baishuwen/Desktop/mlrl_project/results:/results \
  -v /Users/baishuwen/Desktop/mlrl_project/src/mjpro131:/root/.mujoco/mjpro131 \
  -v /Users/baishuwen/Desktop/mlrl_project/src/mjkey.txt:/root/.mujoco/mjkey.txt \
  rllab-adv

pip install -e /tmp/gym-adv

pip install 'mujoco_py<0.5.8' --no-deps
pip install 'imageio<2.1.0' 'pillow<8.0.0' 'numpy<1.16' 'cython' 'lockfile'
