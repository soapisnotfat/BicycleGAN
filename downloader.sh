FILE=$1
URL=https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/$FILE.tar.gz
TAR_FILE=$FILE.tar.gz
TARGET_DIR=dataset/
wget -N $URL -O $TAR_FILE
mkdir $TARGET_DIR
tar -zxvf $TAR_FILE -C $TARGET_DIR
rm $TAR_FILE