import subprocess
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, required=True,
                        help='Language of the wikipedia dump')
    parser.add_argument('--date_1', type=str, required=True,
                        help='Date of the 1st wikipedia dump in the format YYYYMMDD')
    parser.add_argument('--date_2', type=str, required=True,
                        help='Date of the 2nd wikipedia dump in the format YYYYMMDD')
    parser.add_argument('--date_3', type=str, required=True,
                        help='Date of the 3rd wikipedia dump in the format YYYYMMDD')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to directory where to read and write relevant sub-directories')
    parser.add_argument('--processes', type=int, required=True,
                        help='Number of processes to use when using multiprocessing')
    parser.add_argument('--no_mask_perc', type=float, default=0.4,
                        help='Percentage of links where no masking is applied')
    parser.add_argument('--mask_mention_perc', type=float, default=0.2,
                        help='Percentage of links where mention masking is applied')
    parser.add_argument('--mask_sentence_perc', type=float, default=0.3,
                        help='Percentage of links where sentence masking is applied')
    parser.add_argument('--mask_paragraph_perc', type=float, default=0.1,
                        help='Percentage of links where paragraph masking is applied')
    parser.add_argument('--download_processes', type=int, default=5,
                        help='Number of processes to use when downloading Wikipedia HTML pages directly')
    parser.add_argument('--neg_samples_train', nargs='+', type=int, required=True,
                        help='Number of negative samples for training in each stage')
    parser.add_argument('--neg_samples_val', nargs='+', type=int, required=True,
                        help='Number of negative samples for evaluation in each stage')
    parser.add_argument('--max_train_samples', nargs='+', type=int,
                        required=True, help='Maximum number of training samples in each stage')
    parser.add_argument('--max_val_samples', nargs='+', type=int, required=True,
                        help='Maximum number of evaluation samples in each stage')
    parser.add_argument('--neg_strategies', nargs='+', type=str, required=True, help='Negative sampling strategies: 1) replace source with random source not connected to target, 2) replace source with random source connected to target, 3) replace target with random target not connected to source, 4) replace target with random target connected to source, 5) replace context with random context, 6) replace context with context from the same source page')

    args = parser.parse_args()
    
    # run page extractor script for all 3 dates
    print('Running page extractor script...')
    for date in [args.date_1, args.date_2, args.date_3]:
        print(f'... for date {date}')
        subprocess.run(['python',
                        'pages_extractor.py',
                        '--input_dir', os.path.join(args.data_dir, f'{args.lang}wiki-NS0-{date}', 'raw_data'),
                        '--language', args.lang,
                        '--date', date,
                        '--output_dir', os.path.join(args.data_dir, f'{args.lang}wiki-NS0-{date}', 'processed_data')])
    
    # run link extractor script for all 3 dates 
    print('Running link extractor script...')
    for date in [args.date_1, args.date_2, args.date_3]:
        print(f'... for date {date}')
        subprocess.run(['python',
                        'links_extractor_pool.py',
                        '--input_dir', os.path.join(args.data_dir, f'{args.lang}wiki-NS0-{date}', 'raw_data'),
                        '--page_ids', os.path.join(args.data_dir, f'{args.lang}wiki-NS0-{date}', 'processed_data', 'simple_pages.parquet'),
                        '--redirect_map', os.path.join(args.data_dir, f'{args.lang}wiki-NS0-{date}', 'processed_data', 'redirect_map.parquet'),
                        '--output_dir', os.path.join(args.data_dir, f'{args.lang}wiki-NS0-{date}', 'processed_data'),
                        '--processes', str(args.processes)])
        
    # run the post-processing script for all 3 dates
    print('Running post-processing script...')
    for date in [args.date_1, args.date_2, args.date_3]:
        print(f'... for date {date}')
        subprocess.run(['python',
                        'post_processing.py',
                        '--input_dir', os.path.join(args.data_dir, f'{args.lang}wiki-NS0-{date}', 'processed_data'),
                        '--output_dir', os.path.join(args.data_dir, f'{args.lang}wiki-NS0-{date}', 'processed_data')])
    
    # run the synthetic test data generator script
    print('Running synthetic test data generator script...')
    subprocess.run(['python',
                    'synth_test_data_generator.py',
                    '--first_month_dir', os.path.join(args.data_dir, f'{args.lang}wiki-NS0-{args.date_2}', 'processed_data'),
                    '--second_month_dir', os.path.join(args.data_dir, f'{args.lang}wiki-NS0-{args.date_3}', 'processed_data'),
                    '--output_dir', os.path.join(args.data_dir, f'{args.lang}wiki-NS0-{args.date_2}', 'eval_synth'),
                    '--no_mask_perc', str(args.no_mask_perc),
                    '--mask_mention_perc', str(args.mask_mention_perc),
                    '--mask_sentence_perc', str(args.mask_sentence_perc),
                    '--mask_paragraph_perc', str(args.mask_paragraph_perc)])

    # run the test data versions finder
    print('Running test data versions finder...')
    subprocess.run(['python',
                    'test_data_versions_finder.py',
                    '--raw_data_dir', os.path.join(args.data_dir, f'{args.lang}wiki-NS0-{args.date_3}', 'raw_data'),
                    '--first_month_dir', os.path.join(args.data_dir, f'{args.lang}wiki-NS0-{args.date_2}', 'processed_data'),
                    '--second_month_dir', os.path.join(args.data_dir, f'{args.lang}wiki-NS0-{args.date_3}', 'processed_data'),
                    '--output_dir', os.path.join(args.data_dir, f'{args.lang}wiki-NS0-{args.date_2}', 'eval'),
                    '--lang', args.lang,
                    '--first_date', args.date_2,
                    '--second_date', args.date_3,])
    
    # run the html download prep script
    print('Running html download prep script...')
    subprocess.run(['python',
                    'test_html_downloader_prep.py',
                    '--input_file', os.path.join(args.data_dir, f'{args.lang}wiki-NS0-{args.date_2}', 'eval', 'link_versions.parquet'),
                    '--output_directory', os.path.join(args.data_dir, f'{args.lang}wiki-NS0-{args.date_2}', 'eval', 'pages')])

    # download html pages
    print('Downloading html pages...')
    subprocess.run(['node',
                    'crawler/crawl_wiki.js',
                    '--articles', os.path.join(args.data_dir, f'{args.lang}wiki-NS0-{args.date_2}', 'eval', 'pages.txt'),
                    '--concurrency', str(args.download_processes),
                    '--destinationDirectory', os.path.join(args.data_dir, f'{args.lang}wiki-NS0-{args.date_2}', 'eval', 'pages')])
                    
    