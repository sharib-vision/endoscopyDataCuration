#!/bin/bash

# load your environment
source activate torch37

videoArray=(M_01062017122627_0000000000001219_1_001_001-1.MP4)
# M_15032018123342_0000000000001332_1_001_001-1.MP4
# M_01032017120001_0000000000001603_1_001_001-1.MP4
# M_13042017091412_0000000000001728_1_001_001-1.MP4
# M_12102017085034_0000000000001801_1_001_001-1.MP4
# M_02112017112348_0000000000001911_1_001_001-1.MP4
# M_16032017134720_0000000000002011_1_001_001-1.MP4)

# (linux for windows --> https://www.cygwin.com)

# please use path 
resultFolder=./outputVideos
mkdir -p $resultFolder
####################################################################################
declare -a videoArray
declare -a gpusArray

echo "==========================="
executeVideoProcessing(){

    local vidFile=$1
    local GID=$2
    echo "videoFile id: $vidFile"
    
    # path for video folder (whereever your raw video files are)
    # TODO: wehre are videos are located
    videoPath='/well/rittscher/projects/endoscopy_ALI/Exp_3D_dataset/firstVisitData/videoDataOnly'
    
    # path for your code
    BASE_FOLDER=./endoscopyDataCuration
    # path to checkpoint folder
    CKPT_FOLDER=./endoscopyDataCuration/ckpt
    # 
    resultFolder=$resultFolder
    videoFileNamewithFullPath=$videoPath/$vidFile 
    videoOutput='vid-curated_'$vidFile

    echo '----vid file: ' $videoFileNamewithFullPath
    echo '\n : resultFile: '$resultFolder/$videoOutput
    
    # python $BASEFOLDER/crop_video_and_classify_informative_frames.py --videoFile $DATA_FOLDER/$vidFile/$vidFile/MOVIE/TITLE\ 001/M_$vidFile'_1_001_001-1.MP4' --DNN_model_upperGI $BASEFOLDER/DNN_Models/binaryEndoClassifier_124_124.h5 --Result_dir $RESULT_FOLDER
     python $BASE_FOLDER/crop_video_and_classify_endo_frames.py  \
     --ckpt $CKPT_FOLDER/CNN_network_128x128_positive_samples \
     --videoInputFile $videoFileNamewithFullPath  \
     --videoOutputFolder $resultFolder 
     
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
doParallel=0
paraNumJobs=4      # set this according to the number of gpus you want to use
paraDelay=1          # minimal number of seconds between two job starts
paraNice=10          # nice value
paraTimeOut=36000    # time after a running job is killed: either absolut value in seconds (e.g., =10) or percentage of median runtime (e.g., =1000%)
paraProgress=1       # show progress
startTime=`date +%s`
GPU_ID=('3')
# TO install parallel
# https://gist.github.com/drhirsch/e0295105a36039aa38ce936f39b26301
# GPU_ID=('0' '1' '2' '3')
if [[ $doParallel == 1 ]]
    then
    logFile=log_parallel.txt
    progress="--progress --bar"
    
    # parallel installation 
    parallel --gnu --xapply --joblog $logFile $progress \
    --jobs $paraNumJobs --nice $paraNice --delay $paraDelay --timeOut $paraTimeOut \
executeVideoProcessing ::: ${videoArray[@]} ::: ${GPU_ID[@]}
else
    executeVideoProcessing ::: ${videoArray[@]} ::: ${GPU_ID[@]}

fi

echo "=========================================== xxx DONE xxx ========================================================================="



