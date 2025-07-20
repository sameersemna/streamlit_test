echo "Copying necessary files"

dir_rakuten='../apr25_bds_rakuten_2'
dir_reports='./reports'

copy_file() {
    file_path=$1
    echo "Copying $file_path..."
    cp $dir_rakuten/$file_path ./$file_path
    echo "Success Copied $file_path"
}

copy_file 'data/raw/X_train_update.parquet'
copy_file 'data/raw/X_test_update.parquet'
copy_file 'data/raw/Y_train_CVw08PX.parquet'

cp -r $dir_rakuten/data/processed/weights ./data/processed
mkdir -p $dir_reports
cp -r $dir_rakuten/reports/figures $dir_reports

for filepath in "$dir_reports/figures"/*.gif; do
    # Check if the current item is a regular file (not a directory)
    if [ -f "$filepath" ]; then
        # Get the base filename from the full path (e.g., "my_report.txt" from "/path/to/reports/my_report.txt")
        filename=$(basename "$filepath")

        # Remove the extension from the filename
        # ${filename%.*} removes the shortest match of '.*' from the end of the string.
        filename_without_extension="${filename%.*}"
        filename_without_extension="${filename_without_extension%.*}"

        # Print the filename without its extension
        echo "$filename_without_extension"
        cp $dir_rakuten/data/raw/images/image_train/$filename_without_extension.* $dir_reports/figures/
        mv $dir_reports/figures/$filename_without_extension.png.gif $dir_reports/figures/$filename_without_extension.gif
    fi
done

echo "--- End of list ---"
