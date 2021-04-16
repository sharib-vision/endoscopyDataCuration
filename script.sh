#!/bin/bash

#source activate objDetection

# source activate myenv
source activate torch37

videoArray=(M_01062017122627_0000000000001219_1_001_001-1.MP4)
# M_15032018123342_0000000000001332_1_001_001-1.MP4
# M_01032017120001_0000000000001603_1_001_001-1.MP4
# M_13042017091412_0000000000001728_1_001_001-1.MP4
# M_12102017085034_0000000000001801_1_001_001-1.MP4
# M_02112017112348_0000000000001911_1_001_001-1.MP4
# M_16032017134720_0000000000002011_1_001_001-1.MP4)

resultFolder=/well/rittscher/users/sharib/segmentation/Segmentation_BE_BEJ/68_patients_vidAnalysis_v2
mkdir -p $resultFolder
####################################################################################
declare -a videoArray
declare -a gpusArray

echo "==========================="
executeVideoProcessing(){

    local vidFile=$1
    local GID=$2
    echo "videoFile id: $vidFile"
    videoPath='/well/rittscher/projects/endoscopy_ALI/Exp_3D_dataset/firstVisitData/videoDataOnly'
    BASE_FOLDER=/well/rittscher/users/sharib/segmentation/Segmentation_BE_BEJ/codes_seg_analysis
    CKPT_FOLDER=/well/rittscher/users/sharib/segmentation/Segmentation_BE_BEJ/ckpt
    resultFolder=/well/rittscher/users/sharib/segmentation/Segmentation_BE_BEJ/68_patients_vidAnalysis_v2
    videoFileNamewithFullPath=$videoPath/$vidFile 

    videoOutput='vid-CM_'$videoFileName
    videoOutput='vid-CM_'$vidFile

    echo '----vid file: ' $videoFileNamewithFullPath
    echo '\n : resultFile: '$resultFolder/$videoOutput


# echo 'videoFile taken is: ' $vidFile/MOVIE/"TITLE /001/M_"$vidFile'_1_001_001-1.MP4'

# python $BASEFOLDER/crop_video_and_classify_informative_frames.py --videoFile $DATA_FOLDER/$vidFile/$vidFile/MOVIE/TITLE\ 001/M_$vidFile'_1_001_001-1.MP4' --DNN_model_upperGI $BASEFOLDER/DNN_Models/binaryEndoClassifier_124_124.h5 --Result_dir $RESULT_FOLDER
 python $BASE_FOLDER/convexHull_fit_shape_Barretts_3D.py  \
 -weightFile $CKPT_FOLDER/best_deeplabv3plus_resnet50_voc_os16_BE_BEJ_3D_seg.pth \
 -namesClass $BASE_FOLDER/endo.data -class_names obj_BE.names   \
 -videoFile $videoFileNamewithFullPath \
 -resultFolder $resultFolder \
 -videoOutputFile $videoOutput -GPU_ID $GID
    if [ $? != 0 ]
    then
        echo "ERROR while executing videoCompression"
    exit
    fi
}
export -f executeVideoProcessing

####################################################################################
### Parallel execution using <parallel --gnu>
####################################################################################
doParallel=1
paraNumJobs=1       # set this according to the number of gpus you want to use
paraDelay=1          # minimal number of seconds between two job starts
paraNice=10          # nice value
paraTimeOut=36000    # time after a running job is killed: either absolut value in seconds (e.g., =10) or percentage of median runtime (e.g., =1000%)
paraProgress=1       # show progress
startTime=`date +%s`
GPU_ID=('3')
# GPU_ID=('0' '1' '2' '3')
if [[ $doParallel == 0 ]]
    then
    logFile=log_parallel.txt
    progress="--progress --bar"
    /users/rittscher/sharib/bin/parallel --gnu --xapply --joblog $logFile $progress \
    --jobs $paraNumJobs --nice $paraNice --delay $paraDelay --timeOut $paraTimeOut \
executeVideoProcessing ::: ${videoArray[@]} ::: ${GPU_ID[@]}

fi

echo "=========================================== xxx DONE xxx ========================================================================="



