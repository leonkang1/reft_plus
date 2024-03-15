input="/data1/kangxueze/Datasets/wikioutput/AA/wiki_00"
output_prefix="wikioutput"
vocab_file="/home/kangxueze/reft_plus/examples_deepspeed/data_efficiency/gpt/pretrain/gpt2-vocab.json"
merge_file="/home/kangxueze/reft_plus/examples_deepspeed/data_efficiency/gpt/pretrain/gpt2-merges.txt"


python tools/preprocess_data.py \
       --input ${input} \
       --output-prefix ${output_prefix} \
       --vocab-file ${vocab_file} \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file ${merge_file} \
       --append-eod \
       --workers 20
