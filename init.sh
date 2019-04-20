# Check vritual env? $VIRTUAL_ENV

git submodule update --init --recursive
cd ./Mask_RCNN
echo "Installing Mask RCNN requirements"
pip install -r requirements.txt
echo "Installing Mask RCNN"
python setup.py install
cd ../