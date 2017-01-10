#!/bin/bash

WORK_DIR=$PWD
RAW_DIR="$PWD/data/raw"
ALIGNED_DIR="$PWD/data/aligned"
GAMO_DIR="$PWD/../gamo/data/aligned"
FER_DIR="$PWD/../fer2013/data/aligned64"

test(){
    if [ -d $RESULT_DIR/rep-$1/cife_train ]; then
        python ../evaluation/classify.py --trainDir $RESULT_DIR/rep-$1/cife_train \
            --testDir $RESULT_DIR/rep-$1/cife_test --pathName cife
        if [ "$1" -ge 900 ]; then
            python ../evaluation/classify.py --trainDir $RESULT_DIR/rep-$1/gamo_train \
                --testDir $RESULT_DIR/rep-$1/gamo_test --pathName gamo
            python ../evaluation/classify.py --trainDir $RESULT_DIR/rep-$1/fer2013_train \
                --testDir $RESULT_DIR/rep-$1/fer2013_test --pathName fer2013
        fi
    fi
}

for i in contrastive triplet siamese
do
    for j in 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934 935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971 972 973 974 975 976 977 978 979 980 981 982 983 984 985 986 987 988 989 990 991 992 993 994 995 996 997 998 999 1000
    do
        RESULT_DIR="$WORK_DIR/data/results_$i/nn4.small2"
        test $j
    done
done
