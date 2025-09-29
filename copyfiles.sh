echo "Copying necessary files"

dir_source='../jul25_bmlops_flooding'

# ==============================================================================
# FUNCTION: replace_text_in_file
# ------------------------------------------------------------------------------
# Performs an in-place, global, case-sensitive replacement.
# This version uses standard GNU sed, which is native to Ubuntu/Linux.
# ------------------------------------------------------------------------------
# Parameters:
# $1: filepath  - The path to the file to be modified.
# $2: from_text - The exact string to search for (OLD_TEXT).
# $3: to_text   - The replacement string (NEW_TEXT).
# ==============================================================================
replace_text_in_file() {
    # Check if the correct number of arguments was passed
    if [ "$#" -ne 3 ]; then
        echo "Usage: replace_text_in_file <filepath> <from_text> <to_text>"
        return 1
    fi

    local filepath="$1"
    local from_text="$2"
    local to_text="$3"

    # 1. Check if the file exists
    if [ ! -f "${filepath}" ]; then
        echo "Error: File not found at path: ${filepath}"
        return 1
    fi

    # 2. Perform the in-place substitution using GNU sed
    #
    # We use '|' as the substitution delimiter (s|...|...|g) to avoid issues
    # if the input strings contain slashes '/'.
    # The 's' command is built dynamically to ensure variables are correctly parsed.
    
    echo "Replacing '${from_text}' with '${to_text}' in ${filepath}..."

    # Create the substitution command string: s|OLD_TEXT|NEW_TEXT|g
    local sed_command="s|${from_text}|${to_text}|g"

    # Execute sed in-place (-i) without creating a backup file
    sed -i "${sed_command}" "${filepath}"

    if [ $? -eq 0 ]; then
        echo "✅ Replacement successful."
    else
        echo "❌ Replacement failed. Check input strings or file permissions."
        return 1
    fi
}

copy_file() {
    file_path=$1
    echo "Copying $file_path..."
    cp $dir_source/$file_path ./$file_path
    echo "Success Copied $file_path"
}

copy_file 'Architecture.JPG'
copy_file '.env'

cp $dir_source/requirements_fe.txt ./requirements.txt
cp $dir_source/src/fe/*.py ./
cp -r $dir_source/src/fe/assets ./assets

replace_text_in_file "./constants.py" "DIR_SRC = os.path.join(DIR_FE, '../..')" "DIR_SRC = DIR_FE"

echo "--- End of list ---"
