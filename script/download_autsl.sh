#!/bin/bash
# Use "pv" for display progress while unzipping.
################################################################################

if (( $# < 1 ))
then
	echo "USAGE: $0 <argument1> <argument2> ..."
	exit
fi

_MAIN_FOLDER="./AUTSL"

mkdir -p $_MAIN_FOLDER

#------------------------- Install PV -------------------------#
echo "***** Install dependencies *****"
echo
apt-get install pv

#------------------------- CLASSES -------------------------#
class()
{
    # Configuration
    _CLASSES_URL="https://data.chalearnlap.cvc.uab.cat/AuTSL/data/SignList_ClassId_TR_EN.csv"

    echo "AUTSL Dataset - Clasess"
    echo

    mkdir -p "$_MAIN_FOLDER/classes"

    echo "***** Downloading classes *****"
    echo
    wget -O "$_MAIN_FOLDER/classes/classes.csv" $_CLASSES_URL
}

#------------------------- TRAIN Data -------------------------#

# Configuration
_TRAIN_VIDEO_URL="http://158.109.8.102/AuTSL/data/train/train_set_vfbha39.zip"
_TRAIN_VIDEO_FILE="train_set_vfbha39.zip"
_TRAIN_VIDEO_PASSWORD="MdG3z6Eh1t"
_TRAIN_LABELS_URL="http://158.109.8.102/AuTSL/data/train/train_labels.csv"

train_labels()
{
    echo "***** Downloading Train labels *****"
    echo

    mkdir -p "$_MAIN_FOLDER/train"

    wget -O "$_MAIN_FOLDER/train/train_labels.csv" $_TRAIN_LABELS_URL
}

train()
{
    echo "AUTSL Dataset - TRAIN data with labels"
    echo

    # Videos
    echo "***** Downloading videos (18 parts) *****"
    echo
    wget "$_TRAIN_VIDEO_URL.001"
    wget "$_TRAIN_VIDEO_URL.002"
    wget "$_TRAIN_VIDEO_URL.003"
    wget "$_TRAIN_VIDEO_URL.004"
    wget "$_TRAIN_VIDEO_URL.005"
    wget "$_TRAIN_VIDEO_URL.006"
    wget "$_TRAIN_VIDEO_URL.007"
    wget "$_TRAIN_VIDEO_URL.008"
    wget "$_TRAIN_VIDEO_URL.009"
    wget "$_TRAIN_VIDEO_URL.010"
    wget "$_TRAIN_VIDEO_URL.011"
    wget "$_TRAIN_VIDEO_URL.012"
    wget "$_TRAIN_VIDEO_URL.013"
    wget "$_TRAIN_VIDEO_URL.014"
    wget "$_TRAIN_VIDEO_URL.015"
    wget "$_TRAIN_VIDEO_URL.016"
    wget "$_TRAIN_VIDEO_URL.017"
    wget "$_TRAIN_VIDEO_URL.018"

    echo "***** Merging *****"
    echo
    cat $_TRAIN_VIDEO_FILE* > train_videos.zip

    echo "***** Unzipping *****"
    echo
    unzip -q -P $_TRAIN_VIDEO_PASSWORD -o train_videos.zip -d $_MAIN_FOLDER & pv -p -d "$!"
    echo

    echo "***** Cleaning *****"
    echo
    rm $_TRAIN_VIDEO_FILE*
    rm train_videos.zip

    echo "***** Sorting *****"
    echo
    mkdir -p "./AUTSL/train/color"
    mkdir -p "./AUTSL/train/depth"
    mv ./AUTSL/train/*_color.mp4 "./AUTSL/train/color"
    mv ./AUTSL/train/*_depth.mp4 "./AUTSL/train/depth"

    # Labels
    train_labels
}

#------------------------- TEST Data -------------------------#

# Configuration
_TEST_VIDEO_URL="http://158.109.8.102/AuTSL/data/test/test_set_xsaft57.zip"
_TEST_VIDEO_FILE="test_set_xsaft57.zip"
_TEST_VIDEO_PASSWORD="ds6Kvdus3o"
_TEST_LABELS_URL="http://158.109.8.102/AuTSL/data/test/test_labels.zip"
_TEST_LABELS_PASSWORD="ds6Kvdus3o"

test_labels()
{
    echo "***** Downloading Test labels *****"
    echo

    mkdir -p "$_MAIN_FOLDER/test"

    wget -O "$_MAIN_FOLDER/test/test_labels.zip" $_TEST_LABELS_URL
    
    echo "***** Unzipping labels *****"
    echo
    unzip -q -P $_TEST_LABELS_PASSWORD -o "$_MAIN_FOLDER/test/test_labels.zip" -d "$_MAIN_FOLDER/test"
    mv "$_MAIN_FOLDER/test/ground_truth.csv" "$_MAIN_FOLDER/test/test_labels.csv"
    echo

    echo "***** Cleaning labels *****"
    echo
    rm "$_MAIN_FOLDER/test/test_labels.zip"
}

test()
{
    echo "AUTSL Dataset - TEST data with labels"
    echo

    # Videos
    echo "***** Downloading videos (3 parts) *****"
    echo
    wget "$_TEST_VIDEO_URL.001"
    wget "$_TEST_VIDEO_URL.002"
    wget "$_TEST_VIDEO_URL.003"

    echo "***** Merging *****"
    echo
    cat $_TEST_VIDEO_FILE* > test_videos.zip

    echo "***** Unzipping *****"
    echo
    unzip -q -P $_TEST_VIDEO_PASSWORD -o test_videos.zip -d $_MAIN_FOLDER & pv -p -d "$!"
    echo

    echo "***** Cleaning *****"
    echo
    rm $_TEST_VIDEO_FILE*
    rm test_videos.zip

    echo "***** Sorting *****"
    echo
    mkdir -p "./AUTSL/test/color"
    mkdir -p "./AUTSL/test/depth"
    mv ./AUTSL/test/*_color.mp4 "./AUTSL/test/color"
    mv ./AUTSL/test/*_depth.mp4 "./AUTSL/test/depth"

    # Labels
    test_labels
}

#------------------------- VALIDATION Data -------------------------#

# Configuration
_VAL_VIDEO_URL="http://158.109.8.102/AuTSL/data/validation/val_set_bjhfy68.zip"
_VAL_VIDEO_FILE="val_set_bjhfy68.zip"
_VAL_VIDEO_PASSWORD="bhRY5B9zS2"
_VAL_LABELS_URL="http://158.109.8.102/AuTSL/data/validation/validation_labels.zip"
_VAL_LABELS_PASSWORD="zYX5W7fZ"

validation_labels()
{
    echo "***** Downloading Validation labels *****"
    echo

    mkdir -p "$_MAIN_FOLDER/val"

    wget -O "$_MAIN_FOLDER/val/validation_labels.zip" $_VAL_LABELS_URL

    echo "***** Unzipping labels *****"
    echo
    unzip -q -P $_VAL_LABELS_PASSWORD -o "$_MAIN_FOLDER/val/validation_labels.zip" -d "$_MAIN_FOLDER/val"
    mv "$_MAIN_FOLDER/val/ground_truth.csv" "$_MAIN_FOLDER/val/validation_labels.csv"
    echo

    echo "***** Cleaning labels *****"
    echo
    rm "$_MAIN_FOLDER/val/validation_labels.zip"
}

validation()
{
    echo "AUTSL Dataset - VALIDATION data with labels"
    echo

    # Videos
    echo "***** Downloading videos (3 parts) *****"
    echo
    wget "$_VAL_VIDEO_URL.001"
    wget "$_VAL_VIDEO_URL.002"
    wget "$_VAL_VIDEO_URL.003"

    echo "***** Merging *****"
    echo
    cat $_VAL_VIDEO_FILE* > validation_videos.zip

    echo "***** Unzipping *****"
    echo
    unzip -q -P $_VAL_VIDEO_PASSWORD -o validation_videos.zip -d $_MAIN_FOLDER & pv -p -d "$!"
    echo

    echo "***** Cleaning *****"
    echo
    rm $_VAL_VIDEO_FILE*
    rm validation_videos.zip

    echo "***** Sorting *****"
    echo
    mkdir -p "./AUTSL/val/color"
    mkdir -p "./AUTSL/val/depth"
    mv ./AUTSL/val/*_color.mp4 "./AUTSL/val/color"
    mv ./AUTSL/val/*_depth.mp4 "./AUTSL/val/depth"

    # Labels
    validation_labels
}

for ARG in "$@"
do
	shift
	case "${ARG}" in
        "class")             class;;
        "train")             train;;
        "train_labels")      train_labels;;
		"test")              test;;
        "test_labels")       test_labels;;
		"validation")        validation;;
        "validation_labels") validation_labels;;
		*) echo "${ARG}: Invalid argument given";;
	esac
done