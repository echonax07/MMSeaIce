module purge
module load python/3.10


cwd=$(pwd)

echo "loading module done"

echo "Creating new virtualenv"

virtualenv ~/$1
source ~/$1/bin/activate

echo "Activating virtual env"

# pip install --no-index --upgrade pip

cd ~
echo "Downloading Pytorch 2 and Torch Vision"
mkdir -p pip_downloads
cd pip_downloads
pip download torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip download torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

echo "Installing Pytorch 2 and Torch Vision"
pip install --no-index --find-links=. torch==2.0.1
pip install --no-index --find-links=. torchvision==0.15.2

echo "Installing Requirements"
cd $cwd
pip install -r requirements.txt
