#!/bin/bash

imgDim=64

WORK_DIR=$PWD
NOT_ALIGNED_DIR="notaligned${imgDim}"
ALIGNED_DIR="aligned${imgDim}"

test ()
{

   if [ -f $RESULT_DIR/model_$1.t7 ] && [ ! -d $RESULT_DIR/rep-$1/${DATA_LABEL}_test ]; then

        ../batch-represent/main.lua -batchSize 100 -model $RESULT_DIR/model_$1.t7   \
            -data $LABELED_DATA_DIR/test -outDir $RESULT_DIR/rep-$1/${DATA_LABEL}_test -imgDim $imgDim -channelSize 3 $2

        python ../evaluation/classify.py --trainDir $RESULT_DIR/rep-$1/${DATA_LABEL}_train \
                --testDir $RESULT_DIR/rep-$1/${DATA_LABEL}_test --pathName ${DATA_LABEL}
   fi

}


for DATA_DIR in $NOT_ALIGNED_DIR $ALIGNED_DIR
do
    for DATA_LABEL in cife #gamo #fer2013
    do
        LABELED_DATA_DIR="$PWD/../${DATA_LABEL}/data/$DATA_DIR"
        for embSize in 32
        do
            for MODEL_NAME in  "alexnet" "alexnet.v2" "vgg-face" "nn2" "nn4" "nn4.small1" "nn4.small2"
            do
                for i in triplet siamese
                do
                    for j in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250
                    do
                        RESULT_DIR="$WORK_DIR/results/${DATA_DIR}_${embSize}/$i/$MODEL_NAME"

                        test $j "-removeLast 0"
                    done
                done
                for i in contrastive center
                do
                    for j in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250
                    do
                        RESULT_DIR="$WORK_DIR/results/${DATA_DIR}_${embSize}/$i/$MODEL_NAME"

                        test $j "-removeLast 1"
                    done
                done
            done
        done
    done
done