
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

yum install git-lfs
git lfs install
git clone https://huggingface.co/fnlp/moss-moon-003-sft-plugin-int4 # download MOSS 003 from huggingface hub then modify the path in server.py

bash docker_run.sh
