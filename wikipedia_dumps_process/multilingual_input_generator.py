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
    parser.add_argument('--del_current_links', action='store_true', help='Delete the current links from the data')
    
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
            if args.del_current_links:
                for column in df:
                    if 'current_links' in column:
                        df = df.drop(column, axis=1)
            dfs.append(df)
        df = pd.concat(dfs).reset_index(drop=True)
        
        if categories[category] and categories[category] < len(df):
            if args.sampling_strategy == 'uniform':
                df = df.sample(categories[category])
            else:
                freqs = {}
                total = categories[category]
                n_langs = len(args.langs)
                new_total = total
                divide_langs = len(args.langs)
                changed = True
                while changed:
                    changed = False
                    for lang in args.langs:
                        if lang not in freqs and lengths[lang] < total // n_langs:
                            print(f'Not enough samples for language {lang}, using {lengths[lang]} samples instead of {total // n_langs}')
                            freqs[lang] = lengths[lang]
                            new_total -= lengths[lang]
                            divide_langs -= 1
                            changed = True
                    total = new_total
                    n_langs = divide_langs
                for lang in args.langs:
                    if lang not in freqs:
                        freqs[lang] = total // divide_langs
                print(f'In the end, we have the following data distribution')
                for lang in args.langs:
                    print(f'\t{lang}: {freqs[lang]}')
                df = df.groupby('lang', as_index=False).apply(lambda x: x.sample(freqs[x.name])).sample(frac=1).reset_index(drop=True)
                print(df)
                
        # save the data in batches
        for i in tqdm(range(0, len(df), 100000)):
            df_batch = df[i:i+100000].reset_index(drop=True)
            df_batch.to_parquet(os.path.join(args.output_dir, category, category + f'_{i}.parquet'))
        
    print('Done')