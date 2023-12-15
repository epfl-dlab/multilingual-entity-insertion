import argparse
import pandas as pd
import os
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--langs', nargs='+', required=True,
                        help='List of languages to process')
    parser.add_argument('--input_dir', required=True,
                        help='Directory containing the input files')
    parser.add_argument('--output_dir', required=True,
                        help='Directory to write the output files')
    parser.add_argument('--stage', choices=['1', '2', 'test'], type=str, required=True,
                        help='Joining data for stage 1 or 2.')
    parser.add_argument('--max_train_samples', type=int, help='Maximum number of training samples to use')
    parser.add_argument('--max_val_samples', type=int, help='Maximum number of validation samples to use')
    parser.add_argument('--max_test_samples', type=int, help='Maximum number of test samples to use')
    parser.add_argument('--sampling_strategy', choices=['uniform', 'weighted'], default='uniform', help='Sampling strategy to use')
    
    args = parser.parse_args()
    
    # check if the input directory exists
    if not os.path.exists(args.input_dir):
        raise Exception('Input directory does not exist')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
        
    if args.stage == '1':
        categories = {'train': args.max_train_samples, 'val': args.max_val_samples}
    elif args.stage == '2':
        categories = {'train': args.max_train_samples, 'val': args.max_val_samples}
    elif args.stage == 'test':
        categories = {'test': args.max_test_samples}
    
    for category in categories:
        if not os.path.exists(os.path.join(args.output_dir, category)):
            os.makedirs(os.path.join(args.output_dir, category))
            
    # get all the mentions
    dfs = []
    for lang in tqdm(args.langs):
        dfs.append(pd.read_parquet(os.path.join(args.input_dir, lang, 'mentions.parquet')))
    df_mentions = pd.concat(dfs).reset_index(drop=True)
    df_mentions.to_parquet(os.path.join(args.output_dir, 'mentions.parquet'))
    del df_mentions

    for category in categories:
        print(f"Processing {category} data")
        
        dfs = []
        lengths = {}
        for lang in tqdm(args.langs):
            df = pd.read_parquet(os.path.join(args.input_dir, lang, category))
            df['lang'] = lang
            lengths[lang] = len(df)
            dfs.append(df)
        df = pd.concat(dfs).reset_index(drop=True)
        
        if categories[category] and categories[category] < len(df):
            if args.sampling_strategy == 'uniform':
                df = df.sample(categories[category])
            else:
                freqs = {}
                total = categories[category]
                new_total = total
                divide_langs = len(args.langs)
                for lang in args.langs:
                    if lengths[lang] < total // len(args.langs):
                        print(f'Not enough samples for language {lang}, using {lengths[lang]} samples instead of {total // len(args.langs)}')
                        freqs[lang] = lengths[lang]
                        new_total -= lengths[lang]
                        divide_langs -= 1
                total = new_total
                for lang in args.langs:
                    if lang not in freqs:
                        freqs[lang] = total // divide_langs
                print(f'In the end, we have the following data distribution')
                for lang in args.langs:
                    print(f'\t{lang}: {freqs[lang]}')
                df = df.groupby('lang', as_index=False).apply(lambda x: x.sample(freqs[x.name])).sample(frac=1).reset_index(drop=True)
                print(df)
                
        df.to_parquet(os.path.join(args.output_dir, category, category + '.parquet'))
        
    print('Done')