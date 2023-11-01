from sklearn.model_selection import train_test_split
import os.path
import numpy as np
import csv


class SST2DataProcessor:
    def __init__(self, args):
        self.SST2_data_path = args.project_NLP_path + './dataset/SST-2/'
        self.data_file_list = ["train.tsv", "dev.tsv", "test.tsv"] 
        self.sst2_output_path = args.sst2_path

    def read_data(self, flag):
        if flag == "train":
            idx = 0
        elif flag == "val":
            idx = 0
        else:
            idx = 1

        with open(self.SST2_data_path + self.data_file_list[idx], 'r') as f:
            data = f.read()
        lines = data.split('\n')
        sentence = []
        label = []
        cntnull = 0
        for i, l in enumerate(lines):
            if i == 0:
                continue
            l = l.split('\t')
            if len(l) < 2:
                cntnull = cntnull + 1
                continue
            sentence.append(l[0])
            if l[1] == '0':
                label.append(0)
            else:
                label.append(1)
        #         label.append(int(l[1]))
        # print(flag + ": ", cntnull)
        print(flag + " len: ", len(lines) - 1 - cntnull)
        #     sentence = sentence[:100]
        #     label = label[:100]
        data = {
            'sentence': sentence,
            'label': label
        }
        return data

    def load_data(self, flag):
        if flag == 'train':
            train = self.read_data("train")
            train_test_key = train['sentence']
            train_test_value = train['label']
            tra_key, val_key, tra_value, val_value = \
                train_test_split(train_test_key,
                                 train_test_value,
                                 test_size=0.125)
            train = {
                'sentence': tra_key,
                'label': tra_value
            }
            val = {
                'sentence': val_key,
                'label': val_value
            }
            return train, val
        elif flag == 'test':
            test = self.read_data("test")
            return test

    def data_to_txt(self):
        train, val = self.load_data('train')
        train['sentence'].extend(val['sentence'])
        sentence = train['sentence']
        #         temp = []
        #         for i in sentence:
        #             temp.extend(i.split('.'))
        #         data = '\n'.join(temp)
        data = '\n'.join(sentence)
#         print(data)
        with open(self.sst2_output_path, "w") as f:
            f.write(data)


class IMDBDataProcessor:
    def __init__(self, args):
        self.IMDB_data_path = args.project_NLP_path + './dataset/aclImdb'
        self.IMDB_data_save_dir = args.project_NLP_path + './dataset/aclImdb/data'
        self.train_file = self.IMDB_data_save_dir + '/imdb_train.npz'
        self.val_file = self.IMDB_data_save_dir + '/imdb_val.npz'
        self.test_file = self.IMDB_data_save_dir + '/imdb_test.npz'
        self.imdb_output_path = args.imdb_path

    def get_data(self, data_path):
        pos_files = os.listdir(data_path + '/pos')
        neg_files = os.listdir(data_path + '/neg')
        print(len(pos_files))
        print(len(neg_files))

        pos_all = []
        neg_all = []
        for pf, nf in zip(pos_files, neg_files):
            with open(data_path + '/pos' + '/' + pf, encoding='utf-8') as f:
                s = f.read()
                pos_all.append(s)
            with open(data_path + '/neg' + '/' + nf, encoding='utf-8') as f:
                s = f.read()
                neg_all.append(s)

        X_orig = np.array(pos_all + neg_all)
        Y_orig = np.array([1 for _ in range(len(pos_all))] + [0 for _ in range(len(neg_all))])
        print("X_orig:", X_orig.shape)
        print("Y_orig:", Y_orig.shape)

        return X_orig, Y_orig

    def generate_train_data(self):
        X_orig, Y_orig = self.get_data(self.IMDB_data_path + r'/train')
        X_test, Y__test = self.get_data(self.IMDB_data_path + r'/test')
        X = np.concatenate([X_orig, X_test])
        Y = np.concatenate([Y_orig, Y__test])
        np.random.seed = 1
        random_indexs = np.random.permutation(len(X))
        X = X[random_indexs]
        Y = Y[random_indexs]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)
        # print("train len:", X_train.shape)
        # print("val len:", X_val.shape)
        # print("test len:", X_test.shape)
        np.savez(self.IMDB_data_save_dir + '/imdb_train', x=X_train, y=y_train)
        np.savez(self.IMDB_data_save_dir + '/imdb_test', x=X_test, y=y_test)
        np.savez(self.IMDB_data_save_dir + '/imdb_val', x=X_val, y=y_val)

    def convert(self, data, do_banlace=False):
        sentence = data['x'].tolist()
        label = data['y'].tolist()
        if do_banlace:
            cnt0 = 0
            cnt1 = 0
            zero_se = []
            one_se = []
            zero_ll = []
            one_ll = []
            for ii, i in enumerate(label):
                if i == 0:
                    cnt0 = cnt0 + 1
                    zero_se.append(sentence[ii])
                    zero_ll.append(0)
                else:
                    cnt1 = cnt1 + 1
                    one_se.append(sentence[ii])
                    one_ll.append(0)
            print("oversample before")
            print("cnt0:", cnt0)
            print("cnt1:", cnt1)
            if cnt0 < cnt1:
                # oversample the 0 to match the zero number with the one number
                diff = cnt1 - cnt0
                div = diff // len(zero_se)
                mod = diff % len(zero_se)
                for i in range(div):
                    sentence.extend(zero_se)
                    label.extend(zero_ll)
                for i in range(mod):
                    sentence.extend(zero_se[:mod])
                    label.extend(zero_ll[:mod])
            elif cnt0 > cnt1:
                diff = cnt0 - cnt1
                div = diff // len(one_se)
                mod = diff % len(one_se)
                for i in range(div):
                    sentence.extend(one_se)
                    label.extend(one_ll)
                for i in range(mod):
                    sentence.extend(one_se[:mod])
                    label.extend(one_ll[:mod])

        return {
            'sentence': sentence,
            'label': label
        }

    def load_data(self, flag):
        """
            return dict
        """
        if flag == "train":
            train = np.load(self.train_file)
            val = np.load(self.val_file)
            train = self.convert(train, do_banlace=True)
            val = self.convert(val, do_banlace=False)
            return train, val
        elif flag == 'test':
            test = np.load(self.test_file)
            test = self.convert(test, do_banlace=False)
            return test

    def data_to_txt(self):
        train = np.load(self.train_file)
        val = np.load(self.val_file)
        train = self.convert(train, do_banlace=True)
        val = self.convert(val, do_banlace=False)
        sentence = train['sentence']
        sentence.extend(val['sentence'])
        data = '\n'.join(sentence)
        print(data)
        with open(self.imdb_output_path, "w") as f:
            f.write(data)


class AGNEWSDataProcessor:
    def __init__(self, args=None):
        self.AGNEWS_data_path = args.project_NLP_path + './dataset/ag_news/'

        self.data_file_list = ["train.csv", "test.csv"]  
        self.label_map = {"World": 1, "Sports": 2, "Business": 3, "Sci/Tech": 4}

    def read_data(self, flag):
        if flag == "train":
            idx = 0
        elif flag == "val":
            idx = 0
        else:
            idx = 1

        sentence = []
        label = []
        leng = 0
        cnt = 1
        with open(self.AGNEWS_data_path + self.data_file_list[idx], 'r') as f:
            f_csv = csv.reader(f)
            headers = next(f_csv)
            sentence.append(headers[2])
            leng = leng + len(headers[2])
            l = -1
            if headers[0] == '1':
                l = 0
            elif headers[0] == '2':
                l = 1
            elif headers[0] == '3':
                l = 2
            elif headers[0] == '4':
                l = 3
            label.append(l)
            for row in f_csv:
                l = -1
                if row[0] == '1':
                    l = 0
                elif row[0] == '2':
                    l = 1
                elif row[0] == '3':
                    l = 2
                elif row[0] == '4':
                    l = 3
                label.append(l)
                sentence.append(row[2])
                leng = leng + len(row[2])
                cnt = cnt + 1
        print("mean:", leng / cnt)
        data = {
            "sentence": sentence,
            "label": label
        }
        label_num_cnt = []
        for iii in range(4):
            label_num_cnt.append(0)
        for i in label:
            label_num_cnt[i] = label_num_cnt[i] + 1
        for i, j in enumerate(label_num_cnt):
            print("%d label count: %d" % (i, j))
        return data

    def load_data(self, flag):
        if flag == 'train':
            train = self.read_data("train")
            train_test_key = train['sentence']
            train_test_value = train['label']
            tra_key, val_key, tra_value, val_value = \
                train_test_split(train_test_key,
                                 train_test_value,
                                 test_size=0.2)
            train = {
                'sentence': tra_key,
                'label': tra_value
            }
            val = {
                'sentence': val_key,
                'label': val_value
            }
            return train, val
        elif flag == 'test':
            test = self.read_data("test")
            return test


class HATESPEECHDataProcessor:
    def __init__(self, args=None):
        self.HATESPEECH_data_path = args.project_NLP_path + './dataset/hate_speech/'

        self.data_file_list = ["all_files", "sampled_train", "sampled_test"] 
        self.label_file = 'annotations_metadata.csv'
        self.label_map = {"noHate": 0, "hate": 1}
        self.id_label_map = {}
        with open(self.HATESPEECH_data_path + self.label_file, 'r', encoding='utf-8') as f:
            f_csv = csv.reader(f)
            headers = next(f_csv)
            for row in f_csv:
                if row[4] in self.label_map.keys():
                    self.id_label_map[row[0] + '.txt'] = self.label_map[row[4]]
                else:
                    self.id_label_map[row[0] + '.txt'] = -1

        self.test_data = {}
        test_file_name = []
        test_sentence = []
        test_label = []
        for file_name in os.listdir(self.HATESPEECH_data_path + self.data_file_list[2]):
            test_file_name.append(file_name)
            with open(self.HATESPEECH_data_path + self.data_file_list[2] + '/' + file_name, 'r', encoding='utf-8') as f:
                data = f.readlines()
            test_sentence.append(data[0])
            test_label.append(self.id_label_map[file_name])
        self.test_data = {
            'sentence': test_sentence,
            'label': test_label
        }

        self.train_data = {}
        self.val_data = {}
        train_sentence = []
        train_label = []
        val_sentence = []
        val_label = []
        for file_name in os.listdir(self.HATESPEECH_data_path + self.data_file_list[1]):
            test_file_name.append(file_name)
            with open(self.HATESPEECH_data_path + self.data_file_list[1] + '/' + file_name, 'r', encoding='utf-8') as f:
                data = f.readlines()
            train_sentence.append(data[0])
            train_label.append(self.id_label_map[file_name])
            
        
        train_sentence, val_sentence, train_label, val_label = \
                        train_test_split(train_sentence,
                                     train_label,
                                     test_size=0.1)
        self.train_data = {
            'sentence': train_sentence,
            'label': train_label
        }
        self.val_data = {
            'sentence': val_sentence,
            'label': val_label
        }

    def load_data(self, flag):
        if flag == 'train':
            return self.train_data, self.val_data
        elif flag == 'test':
            return self.test_data


if __name__ == '__main__':
    a,b = HATESPEECHDataProcessor().load_data('train')
    print(a)