from .bases import BaseImageDataset
import os.path as osp
import glob
import re
import os


class UnifiedLoader(BaseImageDataset):
    """
    This must be self set

    can be revised in the next version

    verbose: print additional messages
    """
    def __init__(self, dataset_name='Aircraft', data_dir='data_dir', verbose=True):
        super(UnifiedLoader, self).__init__()

        self.root = '/media/space/ZYF/Dataset/Other'

        if dataset_name == 'Aircraft':
            data_dir = '/media/space/ZYF/Dataset/Other/Aircraft/fgvc-aircraft-2013b/data/images'
            train_txt_file = open(os.path.join(self.root, 'aircraft_train.txt'))
            test_txt_file = open(os.path.join(self.root, 'aircraft_test.txt'))
        elif dataset_name == 'StanfordCars':
            data_dir = '/media/space/ZYF/Dataset/Other/StanfordCars/'
            train_txt_file = open(os.path.join(self.root, 'car_train.txt'))
            test_txt_file = open(os.path.join(self.root, 'car_test.txt'))

        self.dataset_dir = data_dir

        data_len = None

        test_img_list = []

        train_img_list = []

        train_label_list = []
        test_label_list = []

        for line in train_txt_file:
            train_img_list.append(line[:-1].split(' ')[0])
            train_label_list.append(int(line[:-1].split(' ')[-1]))

        for line in test_txt_file:
            test_img_list.append(line[:-1].split(' ')[0])
            test_label_list.append(int(line[:-1].split(' ')[-1]))


        dataset_train = []
        dataset_test = []


        # retaining the dataset format
        train_cluster = train_label_list
        test_cluster = test_label_list

        if dataset_name == 'StanfordCars':
            for idx in range(len(train_label_list)):
                imgpath = os.path.join(data_dir,'cars_train', train_img_list[idx])
                dataset_train.append((imgpath, train_label_list[idx]))

            for idx in range(len(test_label_list)):
                imgpath = os.path.join(data_dir,'cars_test', test_img_list[idx])
                dataset_test.append((imgpath, test_label_list[idx]))
        elif dataset_name == 'Aircraft' :
            for idx in range(len(train_label_list)):
                imgpath = os.path.join(data_dir, train_img_list[idx])
                dataset_train.append((imgpath, train_label_list[idx]))

            for idx in range(len(test_label_list)):
                imgpath = os.path.join(data_dir, test_img_list[idx])
                dataset_test.append((imgpath, test_label_list[idx]))
        else:
            assert ("Unknown dataset name!")

        self.train = dataset_train
        self.test = dataset_test

        self.num_train_pids, self.num_train_imgs = self.get_imagedata_info(self.train)
        self.num_test_pids, self.num_test_imgs = self.get_imagedata_info(self.test)

        if verbose:
            print("successful load dataset!!"+dataset_name+'\n')

            self.print_dataset_statistics(self.train,self.test)


    def _process_dir_old(self, data_dir, relabel=True):
        img_paths = glob.glob(osp.join(data_dir, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
