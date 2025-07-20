echo "Copying necessary files"

dir_rakuten='../apr25_bds_rakuten_2'

copy_file() {
    file_path=$1
    echo "Copying $file_path..."
    cp $dir_rakuten/$file_path ./$file_path
    echo "Success Copied $file_path"
}

copy_file 'data/raw/X_train_update.parquet'
copy_file 'data/raw/X_test_update.parquet'
copy_file 'data/raw/Y_train_CVw08PX.parquet'

cp $dir_rakuten/models/weights/image_epochs_3.h5 ./data/processed/image_epochs_3.h5
cp $dir_rakuten/models/weights/image_epochs_3.pkl ./data/processed/image_epochs_3.pkl
