from PIL import Image as Image
from torchvision import transforms as transforms
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

        base_image = Image.open(image_path)
        base_name = os.path.basename(image_path).split('.')[0]

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

    def main_work(self, config_path):
        # load config
        self.load_config(config_path)

        # check output path
        if os.path.exists(self.output_path):
            raise Exception('path "{}" exists !'.format(self.output_path))
            return

        # is file
        if os.path.isfile(self.input_path):
            os.mkdir(self.output_path)
            self.process_single_image(self.input_path, self.output_path)
        
        # is dir
        if os.path.isdir(self.input_path):
            # copy all dirs and files
            shutil.copytree(self.input_path, self.output_path)
            # process all files
            for root, dirs, files in os.walk(self.output_path):
                for file_name in files:
                    self.process_single_image(
                        os.path.join(root, file_name),
                        root
                    )

if __name__ == "__main__":
    IDAT()(sys.argv[1])
