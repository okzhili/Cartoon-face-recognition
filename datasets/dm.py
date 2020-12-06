import glob
import re
import os.path as osp

from .bases import BaseImageDataset
import numpy as np

class DM(BaseImageDataset):

    dataset_dir = ''

    def __init__(self, root='../data', verbose=True, **kwargs):
        super(DM, self).__init__()
        root='images'
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.test_dir = osp.join(self.dataset_dir, 'test')

        train = self._process_dir(self.train_dir, relabel=True)
        query,gallery = self._process_dir_(self.test_dir, relabel=False)

        if verbose:
            print("=> VeRi-776 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)


    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*/*.jpg'))
        pid_container = set()
        for img_path in img_paths:
            pid = img_path.split('/')[-2]
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            c_p = img_path.split('/')[-1][0]
            cp=0
            if c_p=='C':
                cp=0
            else:
                cp=1
            pid = img_path.split('/')[-2]
            camid =-1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid,cp))

        return dataset
    def _process_dir_(self, dir_path, relabel=False):
        # img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        query = np.loadtxt('images/FR_Probe_C2P.txt',dtype='str')
        gallery = np.loadtxt('images/FR_Gallery_C2P.txt',dtype='str')
        query_data = []
        gallery_data=[]
        for queryid in query:
            pid = -1
            camid = queryid
            img_path='images/test/test/'+str(camid)+'.jpg'
            query_data.append((img_path, pid, camid, 1))
        for galleryid in gallery:
            pid = -1
            camid = galleryid
            img_path='images/test/test/'+str(camid)+'.jpg'
            gallery_data.append((img_path, pid, camid, 1))
        return query_data,gallery_data