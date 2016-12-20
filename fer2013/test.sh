#!/usr/bin/env bash

WORK_DIR=$PWD
RAW_DIR="$PWD/data/raw"
ALIGNED_DIR="$PWD/data/aligned"


test ()
{
    if [ -f $RESULT_DIR/model_$1.t7 ] && [ ! -d $RESULT_DIR/rep-$1/train ]; then
        ../batch-represent/main.lua -batchSize 200 -model $RESULT_DIR/model_$1.t7 \
            -data $ALIGNED_DIR/train -outDir $RESULT_DIR/rep-$1/train -imgDim 64 -imgDim 48 -channelSize 1 $2 -cuda

        ../batch-represent/main.lua -batchSize 200 -model $RESULT_DIR/model_$1.t7 \
            -data $ALIGNED_DIR/test -outDir $RESULT_DIR/rep-$1/test -imgDim 64 -imgDim 48 -channelSize 1 $2 -cuda
   fi

   if [ -d $RESULT_DIR/rep-$1/train ]; then
        python ../evaluation/classify.py --trainDir $RESULT_DIR/rep-$1/train \
        --testDir $RESULT_DIR/rep-$1/test
   fi
}


for i in l1hinge cosine triplet
do
    for j in 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934 935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971 972 973 974 975 976 977 978 979 980 981 982 983 984 985 986 987 988 989 990 991 992 993 994 995 996 997 998 999 1000
    do
        RESULT_DIR="$WORK_DIR/data/results_$i/nn4.small2"

         test $j "-removeLast 0"

    done
    RESULT_DIR="$WORK_DIR/data/results_$i/nn4.small2/"

    if [ -d $RESULT_DIR ];then
        python ../util/create_table.py --workDir $RESULT_DIR --title "FER2013_nn4.small2"
    fi

done


for i in loglikelihood
do
    for j in 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934 935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971 972 973 974 975 976 977 978 979 980 981 982 983 984 985 986 987 988 989 990 991 992 993 994 995 996 997 998 999 1000
    do
        RESULT_DIR="$WORK_DIR/data/results_$i/nn4.small2"

         test $j "-removeLast 1"

    done
    RESULT_DIR="$WORK_DIR/data/results_$i/nn4.small2/"

    if [ -d $RESULT_DIR ];then
        python ../util/create_table.py --workDir $RESULT_DIR --title "FER2013_nn4.small2"
    fi

done
