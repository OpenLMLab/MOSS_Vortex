
name="moss_vortex"

batch_size=8
port=7003

docker run -d --rm --name=${name} --privileged --cap-add=SYS_PTRACE --shm-size=500g \
--gpus=all \
-w /mosec -p${port}:${port} \
-v `pwd`:/mosec piglake/mosec:0.6 \
python3 mosec_server.py --port ${port} --timeout 60000  --wait 500 --batches ${batch_size} # --debug


