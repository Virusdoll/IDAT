from PIL import Image as Image
from torchvision import transforms as transforms
import shutil
import json
import os
import sys
import time

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
        # 待保存数据列表(即处理完成的数据列表)
        self.result_list = []
        # 进行一次tf处理后的暂存数据列表
        self.tmp_list = []
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
  
    def process_single_image(self, image_path, save_path):

        base_image = Image.open(image_path).convert('RGB')
        base_name = os.path.basename(image_path).split('.')[0]

        max_length = base_image.size[0]
        if base_image.size[1] > max_length:
            max_length = base_image.size[1]
        if max_length > self.resize_limit:
            resize_p = self.resize_limit / max_length
            base_image = transforms.Resize((
                int(base_image.size[1] * resize_p),
                int(base_image.size[0] * resize_p)
            ))(base_image)

        self.result_list.append({
            'image': base_image,
            'name': base_name
        })

        # 处理任务列表
        for parallel_job in self.job_list:
            # 处理并列任务
            for job in parallel_job:
                job_times = job.get('times')
                # 处理结果集中的每张图片
                for result in self.result_list:
                    tmp_image = result.get('image')
                    tmp_name = result.get('name')
                    # 对每张图片进行对应次数的处理
                    for times in range(job_times):
                        self.tmp_list.append({
                            'image': job.get('tf')(tmp_image),
                            'name': '{}_{}'.format(tmp_name, job.get('func'))
                        })
            self.result_list.extend(self.tmp_list)
            self.tmp_list = []
        
        # delete base image
        os.remove(image_path)

        # 保存结果
        for result in self.result_list:
            result.get('image').save(
                os.path.join(
                    save_path,
                    '{}.jpg'.format(result.get('name'))
                )
            )

        # 清空list
        self.result_list = []

    def seconds_to_time(self, seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return [h, m, s]

    def main_work(self, config_path):
        # load config
        print('load config')
        self.load_config(config_path)

        # check output path
        if os.path.exists(self.output_path):
            raise Exception('path "{}" exists !'.format(self.output_path))

        is_file = os.path.isfile(self.input_path)
        is_dir = os.path.isdir(self.input_path)

        if not (is_file or is_dir):
            raise Exception('input path error')
        
        # copy all dirs and files
        print('copy files and dirs')
        
        # is file
        if is_file:
            os.mkdir(self.output_path)
            shutil.copy(self.input_path, self.output_path)
        
        # is dir
        if is_dir:    
            shutil.copytree(self.input_path, self.output_path)

        # count image number
        total = 0
        for root, dirs, files in os.walk(self.output_path):
            total += len(files)

        # process all files
        print('start processing')
        file_path_list = []
        
        for root, dirs, files in os.walk(self.output_path):
            for file_name in files:
                file_path_list.append([os.path.join(root, file_name), root])
        
        # set count
        count = 0
        
        # start time
        start_time = time.time()
        
        for file_path in file_path_list:
            self.process_single_image(file_path[0], file_path[1])

            count += 1
            cost_time = time.time() - start_time
            eta_time = cost_time * (total - count) / count
            c_h, c_m, c_s = self.seconds_to_time(cost_time)
            e_h, e_m, e_s = self.seconds_to_time(eta_time)
            print('\rfinish {}/{}({:.2f}%), cost {:.0f}:{:.0f}:{:.2f}, eta {:.0f}:{:.0f}:{:.2f}'.format(
                count, total, count/total*100, c_h, c_m, c_s, e_h, e_m, e_s
            ), end='')
        
        print('\nfinish')

if __name__ == "__main__":
    IDAT()(sys.argv[1])
