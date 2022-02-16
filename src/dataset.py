import utils
from torch.utils.data import Dataset
class TrainDataset(Dataset):
    def __init__(self ,cfg , df):
        self.cfg = cfg
        self.feature_texts = df['feature_text'].values
        self.pn_historys = df['pn_history'].values
        self.annotation_legths = df['annotation_length'].values
        self.locations = df['location'].values

    def __len__(self):
        return len(self.feature_texts)

    def __getitem__(self,item):
        inputs = utils.prepare_input(self.cfg ,
                self.pn_historys[item] ,
                self.feature_texts[item])

        label = utils.create_label(self.cfg ,
                self.pn_historys[item] ,
                self.annotation_legths[item] ,
                self.locations[item])

        return inputs , label
