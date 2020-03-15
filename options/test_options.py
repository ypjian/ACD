import argparse
import os.path as osp
class TestOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description="adaptive segmentation netowork")
        parser.add_argument("--model", type=str, default='DeepLab',help="available options : DeepLab and VGG")
        parser.add_argument("--data-dir-target", type=str, default='./../../../data/sunyuze/Cityscapes/data', help="Path to the directory containing the source dataset.")
        parser.add_argument("--data-list-target", type=str, default='./dataset/cityscapes_list/train.txt', help="Path to the file listing the images in the source dataset.")
        parser.add_argument("--data-label-folder-target", type=str, default=None, help="Path to the soft assignments in the target dataset.") 
        parser.add_argument("--num-classes", type=int, default=19, help="Number of classes for cityscapes.")
        parser.add_argument("--init-weights", type=str, default=None, help="initial model.")
        parser.add_argument("--restore-from", type=str, default=None, help="Where restore model parameters from.")
        parser.add_argument("--set", type=str, default='train', help="choose adaptation set.")  
        parser.add_argument("--save", type=str, default='./../../../data2/SYN/sslzuixin', help="Path to save result.")    
        parser.add_argument('--gt_dir', type=str, default = './../../../data/sunyuze/Cityscapes/data', help='directory which stores CityScapes val gt images')
        parser.add_argument('--devkit_dir', default='./dataset/cityscapes_list', help='base directory of cityscapes')         
        return parser.parse_args()
    
   
