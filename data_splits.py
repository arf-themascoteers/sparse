class DataSplits:
    def __init__(self,
                 name,
                 bs_train_x, bs_train_y,
                 evaluation_train_x, evaluation_train_y,
                 evaluation_test_x, evaluation_test_y):
        self.name = name
        self.bs_train_x = bs_train_x
        self.bs_train_y = bs_train_y
        self.evaluation_train_x = evaluation_train_x
        self.evaluation_train_y = evaluation_train_y
        self.evaluation_test_x = evaluation_test_x
        self.evaluation_test_y = evaluation_test_y

    def get_name(self):
        return self.name

    def splits_description(self, short=True):
        desc = f"train={len(self.bs_train_y)};" \
               f"evaluation_train={len(self.evaluation_train_y)};" \
               f"evaluation_test={len(self.evaluation_test_y)};\n"
        if not short:
            desc = f"{desc}bs_train_x={self.bs_train_x[0:3,0]};bs_train_y={self.bs_train_y[0:3]};\n"
            desc = f"{desc}evaluation_train_x={self.evaluation_train_x[0:3,0]};evaluation_train_y={self.evaluation_train_y[0:3]};\n"
            desc = f"{desc}evaluation_test_x={self.evaluation_test_x[0:3,0]};evaluation_test_y={self.evaluation_test_y[0:3]};\n"
        return desc