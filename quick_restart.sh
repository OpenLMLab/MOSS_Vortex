echo "Rudely quick restarting"
echo "Try to stop moss_003_mosec_fp16_with_past (aka 'MOSS 003 Vortex') "
docker stop moss_vortex

echo "wait util docker daemon clean "
sleep 10
echo "running container"
bash docker_run.sh
sleep 40