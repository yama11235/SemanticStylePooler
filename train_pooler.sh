#!/bin/bash
# Todo
source ../my-project/.venv/bin/activate
# comand line arguments, save path, wandb
model=${MODEL:-mixedbread-ai/mxbai-embed-large-v1}  # pre-trained model
pooler_type=${POOLER_TYPE:-avg}  # avg, cls, mean, last
max_seq_length=${MAX_SEQ_LENGTH:-512}  # maximum sequence length for the model
start_layer=${START_LAYER:-23}  # start layer number for the encoder
layer_num=${LAYER_NUM:-23}  # layer number to use for the encoder
use_flash_attention=${USE_FLASH_ATTENTION:-eager}  # whether to use flash attention
fp16=${FP16:-true}  # whether to use fp16
bf16=${BF16:-false}  # whether to use bf16
device_map=${DEVICE_MAP:-cuda}  # device map for the model
lr=${LR:-1e-4}  # learning rate

# model=${MODEL:-Qwen/Qwen3-Embedding-8B}  # pre-trained model
# pooler_type=${POOLER_TYPE:-last}  # avg, cls, mean, last
# max_seq_length=${MAX_SEQ_LENGTH:-2048}  # maximum sequence length for the model
# start_layer=${START_LAYER:-10}  # start layer number for the encoder
# layer_num=${LAYER_NUM:-35}  # layer number to use for the encoder
# use_flash_attention=${USE_FLASH_ATTENTION:-eager}
# fp16=${FP16:-false}
# bf16=${BF16:-true} # whether to use flash attention
# device_map=${DEVICE_MAP:-cuda}
# lr=${LR:-1e-4}  # learning rate

encoding=${ENCODER_TYPE:-bi_encoder}  # cross_encoder, bi_encoder, tri_encoder  # learning rate
# wd=${WD:-0.1}  # weight decay
transform=${TRANSFORM:-False}  # whether to use an additional linear layer after the encoder
# seed=${SEED:-42}
output_dir=${OUTPUT_DIR:-output_train_encoder_full}  # output directory for the model

train_data_size=${TRAIN_DATA_SIZE:-5000}  # number of training samples
# config=clf_data:${train_file_name}_lr:${lr}_wd:${wd}_seed:${seed}

# データセットと対応するラベル名のペアを定義
sets=(
  "stsb stsb_score"
  "sick sick_score"
  "Opusparcus opusparcus_score"
  "CxC cxc_score"
  "STS3k sts3k_score"
  "ArgPairs argpairs_score"
  "BWS bws_score"
  "FinSTS finsts_score"
  "SemRel2024 SemRel2024_score"
  "APT apt_label"
  "PARADE parade_label"
  "Webis-CPC-11 webis_cpc_label"
  "AskUbuntu askubuntu_label"
  "PAWS-Wiki paws_wiki_label"
  "QQP qqp_label"
  "ELSA_joy-sadness-anger-surprise-love-fear emotion"
  "ELSA_conversational-poetic-formal-narrative style"
  "Paradetox_neutral-toxic toxic"
  "APPDIA_neutral-offensive offensive"
  "WNC_subjective-objective objectivity"
  "MTST_positive-negative negative"
)

margin_sets=(0.2)
alpha_sets=(1.0)
dimension_sets=(256)  # 出力次元数のセット
# seed_sets=(42 43 44 45 46)  # シード値のセット
seed_sets=(42)  # シード値のセット
# lr_sets=(1e-2 1e-3 1e-4 1e-5 1e-6)  # 学習率のセット
lr_sets=(1e-4)  # 学習率のセット

for layer in $(seq "$start_layer" "$layer_num"); do
  echo "Processing layer: $layer"
  for lr in "${lr_sets[@]}"; do
    for seed in "${seed_sets[@]}"; do
      for dimension in "${dimension_sets[@]}"; do
        for margin in "${margin_sets[@]}"; do
          for alpha in "${alpha_sets[@]}"; do
            # ループで各ペアを処理
            for pair in "${sets[@]}"; do
              # 各ペアからtrain_file_nameとlabel_nameに分割代入
              read train_file_name label_name <<< "$pair"

              # ファイル名の構築
              train_file=data_preprocessed/${train_file_name}_train.csv
              eval_file=data_preprocessed/${train_file_name}_test.csv
              test_file=data_preprocessed/${train_file_name}_test.csv

              # 必要な処理を書く（ここではechoで表示）
              echo "Processing dataset: $train_file_name"
              echo "  Label column: $label_name"
              echo "  Train file: $train_file"
              echo "  Test file: $test_file"


              for layer in $(seq "$start_layer" "$layer_num"); do
                echo "layer=$layer"
                # config=lr:${lr}_seed:${seed}_layer:${layer}_margin:${margin}_alpha:${alpha}
                config="output_dim-${dimension}_lr-${lr}_layer-${layer}_seed-${seed}"
                wandb_project_name=${config}
                # wandb_project="BERT_hyperparam_search"
                wandb_project="classifier_${train_file_name}"

                  # objectiveの設定
                if [[ "$label_name" == *label* ]]; then
                  objective="binary_classification"
                  type="linear"

                    # classifier_configs をヒアドキュメントで組み立て
                  read -r -d '' classifier_configs <<EOF
{
  "${label_name}": {"type":"${type}","objective":"${objective}","distance":"cosine","output_dim":${dimension},"dropout":0.1,"layer": $layer}
}
EOF

                elif [[ "$label_name" == *score* ]]; then
                  objective="regression"
                  type="linear"

                    # classifier_configs をヒアドキュメントで組み立て
                  read -r -d '' classifier_configs <<EOF
{
  "${label_name}": {"type":"${type}","objective":"${objective}","distance":"cosine","output_dim":256,"dropout":0.1,"layer": $layer}
}
EOF

                else
                  objective="contrastive_logit"
                  type="contrastive_logit"
                  # アンダースコアで分割してラベル部分を取得
                  label_part="${train_file_name#*_}"

                  # ハイフンで分割し、要素数を数えることでラベル数を取得
                  IFS='-' read -ra labels <<< "$label_part"
                  label_count="${#labels[@]}"
                  echo "Label count: $label_count"

                    # classifier_configs をヒアドキュメントで組み立て
                  read -r -d '' classifier_configs <<EOF
{
  "${label_name}": {"type":"${type}","objective":"${objective}","distance":"cosine","intermediate_dim":256,"output_dim":${label_count},"dropout":0.1,"layer": $layer, "margin": ${margin}, "alpha": ${alpha}}
}
EOF

                fi

                config_path="${output_dir}/${model}/${train_file_name}/${config}/${train_file_name}.json"
                mkdir -p "$(dirname "$config_path")"
                echo "$classifier_configs" > "$config_path"

                CUDA_VISIBLE_DEVICES=0 python train_pooler.py \
                  --output_dir "${output_dir}/${model}/${train_file_name}/${config}/" \
                  --classifier_configs "${config_path}" \
                  --model_name_or_path ${model} \
                  --encoding_type ${encoding} \
                  --pooler_type ${pooler_type} \
                  --freeze_encoder False \
                  --max_seq_length ${max_seq_length} \
                  --train_file ${train_file} \
                  --validation_file ${eval_file} \
                  --test_file ${test_file} \
                  --do_train \
                  --do_eval \
                  --do_predict \
                  --eval_strategy "steps" \
                  --eval_steps 50 \
                  --logging_steps 100 \
                  --per_device_train_batch_size 64 \
                  --learning_rate ${lr} \
                  --num_train_epochs 10 \
                  --lr_scheduler_type constant \
                  --warmup_ratio 0.1 \
                  --log_level info \
                  --disable_tqdm False \
                  --save_strategy steps \
                  --save_steps 5000 \
                  --seed ${seed} \
                  --data_seed ${seed} \
                  --fp16 ${fp16} \
                  --bf16 ${bf16} \
                  --log_time_interval 15 \
                  --remove_unused_columns False \
                  --wandb_project_name ${wandb_project_name} \
                  --wandb_project ${wandb_project} \
                  --max_train_samples ${train_data_size} \
                  --max_eval_samples 5000 \
                  --max_predict_samples 5000 \
                  --use_flash_attention ${use_flash_attention} \
                  --overwrite_output_dir True \
                  --device_map ${device_map} \
                  --report_to wandb
                done
              done
            done
          done
        done
      done
    done
  done