import ast
import argparse

def scale_to_range(labels, _min, _max):
    return list(map(lambda x: (x - _min) / (_max - _min), labels))

def parse_dict(dict_str):
    try:
        return ast.literal_eval(dict_str)
    except (ValueError, SyntaxError) as e:
        raise argparse.ArgumentTypeError(f"Invalid dictionary literal: {e}")

def get_preprocessing_function(
        tokenizer,
        sentence1_key,
        sentence2_key,
        sentence3_key,
        aspect_key,
        padding,
        max_seq_length,
        model_args,
        scale=None,
        ):
    if model_args.encoding_type == 'bi_encoder':
        def preprocess_function(examples):
            sent1_args = examples[sentence1_key]
            sent1_result = tokenizer(
                sent1_args,
                padding=padding,
                max_length=max_seq_length,
                truncation=True
            )

            if sentence2_key in examples:
                sent2_args = examples[sentence2_key]
                sent2_result = tokenizer(
                    sent2_args,
                    padding=padding,
                    max_length=max_seq_length,
                    truncation=True
                )
                sent1_result['input_ids_2'] = sent2_result['input_ids']
                sent1_result['attention_mask_2'] = sent2_result['attention_mask']
                if 'token_type_ids' in sent2_result:
                    sent1_result['token_type_ids_2'] = sent2_result['token_type_ids']

            if sentence3_key in examples:
                sent3_args = examples[sentence3_key]
                sent3_result = tokenizer(
                    sent3_args,
                    padding=padding,
                    max_length=max_seq_length,
                    truncation=True
                )
                sent1_result['input_ids_3'] = sent3_result['input_ids']
                sent1_result['attention_mask_3'] = sent3_result['attention_mask']
                if 'token_type_ids' in sent3_result:
                    sent1_result['token_type_ids_3'] = sent3_result['token_type_ids']

            for aspect_name in aspect_key:
                if aspect_name in examples:
                    sent1_result[aspect_name] = examples[aspect_name]
                else:
                    sent1_result[aspect_name] = None

            return sent1_result

    else:
        raise ValueError(f'Invalid model type: {model_args.encoding_type}')

    return preprocess_function