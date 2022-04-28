CUDA_ID=0
ln1="cs"
ln2="en"

CUDA_VISIBLE_DEVICES=${CUDA_ID} python -m src \
    --ln1 ${ln1} \
    --ln2 ${ln2}