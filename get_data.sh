echo "downloading trex data..."

#the code downloads the sample by default, you can change this by commenting the following line and uncommenting the one after
url="https://figshare.com/ndownloader/files/8768701"
#url=https://figshare.com/ndownloader/files/8760241

mkdir data
cd data
wget $url
unzip ${url:39:7} 
rm ${ulr:39:7}

cd ..
echo "done!"
