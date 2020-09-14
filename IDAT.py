from PIL import Image as Image
from torchvision import transforms as transforms
from multiprocessing import cpu_count, Process, Pool, Manager
from time import time, sleep
import shutil
import json
import os
import sys


class MyCrop(object):
    '''
    剪裁比例依赖图片宽度和高度，故需实现动态变更RandomCrop类的实例化参数
    '''
    def __init__(self, w_p, h_p):
        self.w_p = w_p
        self.h_p = h_p
    
    def __call__(self, img):
        img_w = img.size[0]
        img_h = img.size[1]
        crop_w = (int)(self.w_p * img_w)
        crop_h = (int)(self.h_p * img_h)
        output_img = transforms.RandomCrop(size=(crop_h, crop_w))(img)
        return output_img


class IDAT(object):

    def __init__(self):
        # tf处理列表
        self.job_list = None

    def __call__(self, config_path):
        self.main_work(config_path)

    def load_config(self, config_path):
        
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
        
        self.input_path = config.get('input_path')
        self.output_path = config.get('output_path')
        self.resize_limit = config.get('resize_limit')
        self.multi_process = config.get('multi_process')

        if self.multi_process == -1: self.multi_process = cpu_count()
        
        tmp_job_list = []
        for parallel_job in config.get('job'):
            parallel_job_list = []
            for job in parallel_job:
                # config -> function
                parallel_job_list.append(self.choose_job(job))
            tmp_job_list.append(parallel_job_list)

        self.job_list = tmp_job_list

    def choose_job(self, job):
        func = job.get('func')
        times = job.get('times')
        tf = None

        if func == 'h':
            tf = transforms.RandomHorizontalFlip(p=1)
        
        if func == 'c':
            tf = MyCrop(job.get('w_p'), job.get('h_p'))
        
        if func == 'r':
            tf = transforms.RandomRotation(
                degrees=(job.get('min'), job.get('max'))
            )

        if func == 'g':
            tf = transforms.Grayscale(3)

        if func == 'bu' or func == 'bd':
            tf = transforms.ColorJitter(
                brightness=(job.get('min'), job.get('max'))
            )
        
        if func == 'cu' or func == 'cd':
            tf = transforms.ColorJitter(
                contrast=(job.get('min'), job.get('max'))
            )
        
        if func == 'su' or func == 'sd':
            tf = transforms.ColorJitter(
                saturation=(job.get('min'), job.get('max'))
            )

        return {
            'func': func,
            'tf': tf,
            'times': times
        }

    def process_image(self, process_dict, process_n):
        '''
        process images from file_path_list
        '''
        for file_path in self.file_path_list[process_n]:
            self.process_single_image(file_path[0], file_path[1])
            process_dict[process_n] += 1

    def process_single_image(self, image_path, save_path):
        '''
        process one image and save result
        '''

        base_image = Image.open(image_path).convert('RGB')
        base_name = os.path.basename(image_path).split('.')[0]
        result_list = []
        tmp_list = []

        img_w = base_image.size[0]
        img_h = base_image.size[1]
        max_length = img_w if img_w > img_h else img_h

        if max_length > self.resize_limit:
            resize_p = self.resize_limit / max_length
            base_image = transforms.Resize((
                int(base_image.size[1] * resize_p),
                int(base_image.size[0] * resize_p)
            ))(base_image)

        result_list.append({
            'image'     : base_image,
            'name'      : base_name,
            'job_number': 0
        })

        # delete base image
        os.remove(image_path)

        '''
        for less memory cost,
        use algorithm like:
             t(1)  t(2)   t(3)    t(4)     t(5)
        1 -> 1a -> 1ab -> 1abc -> 1abcd -> empty
                -> 1ac -> 1acd
                -> 1ad
          -> 1b -> 1bc -> 1bcd
                -> 1bd
          -> 1c -> 1cd
          -> 1d
        after gen data in t(n),
        data in t(n-1) would be saved to disk and deleted to free memory
        when it can't gen any data(like t(5)),
        algorithm will stop
        '''
        
        job_list_len = len(self.job_list)

        while len(result_list) > 0:
            for result in result_list:
                result_image        = result.get('image')
                result_name         = result.get('name')
                result_job_number   = result.get('job_number')

                result_image.save(
                    os.path.join(save_path, '{}.jpg'.format(result_name))
                )

                result = None

                for parallel_job, job_number in zip(
                    self.job_list,
                    range(1, job_list_len + 1)
                ):
                    if job_number <= result_job_number:
                        continue
                    
                    for job in parallel_job:
                        job_tf      = job.get('tf')
                        job_func    = job.get('func')
                        job_times   = job.get('times')
                        
                        for times in range(job_times):
                            tmp_list.append({
                                'image'     : job_tf(result_image),
                                'name'      : '{}_{}'.format(result_name, job_func),
                                'job_number': job_number
                            })
            
            result_list = tmp_list
            tmp_list = []

    def seconds_to_time(self, seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return [int(h), int(m), int(s)]

    def process_info(self, process_dict):
        start_time = time()
        finish_number = 0
        eta_time = 0
        cost_time = 0
        info_format = (
            '\r  > {:.2f}%({}/{}) ' +
            '| cost {:0>2d}:{:0>2d}:{:0>2d} ' +
            '| eta {:0>2d}:{:0>2d}:{:0>2d}'
        )

        while finish_number < self.file_number:
            new_cost_time = time() - start_time
            
            new_finish_number = 0
            for process_n in range(self.multi_process):
                new_finish_number += process_dict[process_n]

            if new_finish_number == finish_number:
                eta_time = eta_time - (new_cost_time - cost_time)
            
            cost_time = new_cost_time

            if new_finish_number > finish_number:
                finish_number = new_finish_number
                eta_number = self.file_number - finish_number
                eta_time = cost_time / finish_number * eta_number
            
            if eta_time < 0:
                eta_time = 0

            c_h, c_m, c_s = self.seconds_to_time(cost_time)
            e_h, e_m, e_s = self.seconds_to_time(eta_time)

            print(info_format.format(
                finish_number / self.file_number * 100,
                finish_number, self.file_number,
                c_h, c_m, c_s,
                e_h, e_m, e_s
            ), end='')

            sleep(1)

    def main_work(self, config_path):
        '''
        main program of this class
        '''

        # load config
        print('  > load config')
        self.load_config(config_path)

        # check output path
        if os.path.exists(self.output_path):
            raise Exception('output path "{}" exists'.format(self.output_path))

        is_file = os.path.isfile(self.input_path)
        is_dir = os.path.isdir(self.input_path)

        if not (is_file or is_dir):
            raise Exception('input path "{}" error'.format(self.input_path))

        if not self.multi_process > 0:
            raise Exception('process number "{}" error'.format(self.multi_process))
        
        # copy all dirs and files
        print('  > copy files and dirs')
        
        # is file
        if is_file:
            os.mkdir(self.output_path)
            shutil.copy(self.input_path, self.output_path)
        
        # is dir
        if is_dir:    
            shutil.copytree(self.input_path, self.output_path)

        # 分配各线程处理的任务
        self.file_path_list = []
        for process_n in range(self.multi_process):
            self.file_path_list.append([])
        
        process_n = 0
        for root, dirs, files in os.walk(self.output_path):
            for file_name in files:
                self.file_path_list[process_n].append(
                    [os.path.join(root, file_name), root]
                )
                process_n = 0 if process_n == self.multi_process - 1 else process_n + 1
        
        # count file number
        self.file_number = 0
        for file_path in self.file_path_list:
            self.file_number += len(file_path)
        
        # create dict for process communication
        process_dict = Manager().dict()
        for process_n in range(self.multi_process):
            process_dict[process_n] = 0

        # process all files
        print('  > start processing with {} process'.format(self.multi_process))

        # create and run process
        pool = Pool(self.multi_process)
        for process_n in range(self.multi_process):
            pool.apply_async(
                self.process_image,
                args=(process_dict, process_n,)
            )
        pool.close()

        process_info = Process(
            target=self.process_info,
            args=(process_dict,)
        )
        process_info.start()

        pool.join()
        process_info.join()

        # finish
        print('\n  > finish')

if __name__ == "__main__":
    IDAT()(sys.argv[1])
