
#git clone https://github.com/piglaker/vortex.git

if [[ $(which docker) && $(docker --version) ]]; then
    echo "检查到Docker已安装!"
  else
    echo "安装docker环境..."
    curl -fsSL https://get.docker.com | bash -s docker --mirror Aliyun
    echo "安装docker环境...安装完成!"
fi

systemctl start docker

docker login
docker pull piglake/mosec:0.6

bash docker_run.sh
