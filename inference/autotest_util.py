import os
import math
import json
import time
import shutil


def max_wrong_char_for_partial(total_length):
    return math.ceil(total_length * 0.3)


def hamming_distance(str1, str2):
    if len(str1) != len(str2):
        raise ValueError('Длины сравниваемых строк различны')
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))


def recognition_test(pipeline_function, test_data_dir, test_name=None):
    stats = {'total': 0}
    for key in ['equal', 'partial', 'wrong', 'none']:
        stats[key] = {'count': 0}
    failed_lists = {}
    for key in ['partial_images', 'wrong_images', 'not_recognized_images']:
        failed_lists[key] = []

    start_time = time.time()
    for img in os.listdir(test_data_dir):
        img_path = f'{test_data_dir}/{img}'
        if not os.path.isfile(img_path):
            continue
        expected, extension = img.rsplit('.', 1)
        if extension not in ['jpg', 'jpeg', 'png', 'jfif']:
            print(f'Warning: {extension} files are not supported. If they are supported, add them to the list.')
            continue
        
        stats['total'] += 1
        this_max_wrong = max_wrong_char_for_partial(len(expected))
        print(f'Testing {img_path}\nGround truth: {expected}')
        result = pipeline_function(img_path)
        if result == '':
            rank = 'none'
            temp = {'img': img_path, 'ground_truth': expected, 'model_answer': result}
            failed_lists['not_recognized_images'].append(temp)
        elif result == expected:
            rank = 'equal'
        elif len(result) != len(expected) or hamming_distance(result, expected) > this_max_wrong:
            rank = 'wrong'
            temp = {'img': img_path, 'ground_truth': expected, 'model_answer': result}
            failed_lists['wrong_images'].append(temp)
        else:
            rank = 'partial'
            temp = {'img': img_path, 'ground_truth': expected, 'model_answer': result}
            failed_lists['partial_images'].append(temp)
        stats[rank]['count'] += 1
        print(f'Model answer: {result}\nResult: {rank}\n')

    for rank in ['equal', 'partial', 'wrong', 'none']:
        stats[rank]['percent'] = round(stats[rank]['count'] / stats['total'] * 100, 1)

    elapsed_secs = round(time.time() - start_time, 3)
    return {'test_name': test_name,
            'test_folder': os.path.abspath(test_data_dir).replace('\\', '/'),
            'elapsed_seconds': elapsed_secs,
            'stats': stats,
            'failed_lists': failed_lists}


def print_results(test_results):
    stats = test_results['stats']
    print(test_results['test_folder'])
    for i in 'equal', 'partial', 'wrong', 'none':
        print(f'{i.capitalize() + ":": <8} {stats[i]["count"]} / {stats['total']}, {stats[i]["percent"]}%')


def save_results(test_results, results_dir='results'):
    save_dir = f'{results_dir}/test_{time.strftime("%Y%m%d_%H%M%S")}'
    if test_results['test_name'] is not None:
        save_dir += f'_{test_results["test_name"]}'
    os.makedirs(save_dir, exist_ok=True)

    with open(f'{save_dir}/results.json', 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)

    for i in 'partial_images', 'wrong_images', 'not_recognized_images':
        if len(test_results['failed_lists'][i]) > 0:
            os.makedirs(f'{save_dir}/{i}', exist_ok=True)
            for img_data in test_results['failed_lists'][i]:
                name = img_data['img'].replace('/', '_').replace('\\', '_').replace(':', '_')
                shutil.copy2(img_data['img'], f'{save_dir}/{i}/{name}')
