from autotest_util import recognition_test, print_results, save_results
from pipeline import pipeline

if __name__ == '__main__':
    results = recognition_test(pipeline_function=pipeline, test_name='YOLOv11_with_Qwen2-VL-2B-Instruct', test_data_dir='test')
    print_results(results)
    save_results(results)
