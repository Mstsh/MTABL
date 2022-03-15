import keras
import Models as Models
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.constraints import max_norm
from sklearn.metrics import classification_report, f1_score
import pandas as pd
import numpy as np


def classifaction_report_csv(report, location):
    dataframe = pd.DataFrame.from_dict(report)
    dataframe.to_csv(location, index=True)


train_structure = [True, True, True, True, True]
test_set = [1, 1, 1, 1, 1]
f1_cnt = 0
fl_dict = {}

# random data
x = np.random.rand(1000, 40, 10)
y = keras.utils.to_categorical(np.random.randint(0, 3, (1000,)), 3)

# 1 hidden layer network with input: 40x10, hidden 120x5, output 3x1
templates = [[[40, 10], [120, 5], [3, 1]]]

tmplate_cnt = 0
for template in templates:

    Net_Structure = 'TABL_C_2_attention_Con_f_full'

    projection_constraint = max_norm(3.0, axis=0)
    attention_constraint = max_norm(7.0, axis=1)

    main_Net = Net_Structure
    for run_i in range(1, 2):
        Net_Structure = main_Net + '_run_' + str(run_i)
        model = Models.TABL_2attention(template, projection_constraint=projection_constraint,
                                       attention_constraint=attention_constraint, full_TABL=1, L_concatenate=2)
        # create class weight
        # this is a random values, create the class weight based on actual number of class in the train set
        class_weight = {0: 1e6 / 300.0,
                        1: 1e6 / 400.0,
                        2: 1e6 / 300.0}
        # training
        lrate = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20,
                                  # model.layers[2].trainable = False
                                  verbose=0,
                                  mode='min',
                                  min_delta=0.0001, cooldown=0, min_lr=0.00001)
        mode_path = "results/{}.h5".format(Net_Structure)
        checkpoint = ModelCheckpoint(filepath=mode_path, monitor='get_f1', mode='max',
                                     save_best_only=True)
        callbacks_list = [checkpoint, lrate]
        history = model.fit(x, y, batch_size=256, epochs=200, class_weight=class_weight,

                            validation_split=0.15, callbacks=callbacks_list, verbose=1)

        hist_df = pd.DataFrame(history.history)
        hist_df.to_csv('results/History_{}.csv'.format(Net_Structure))
        model.load_weights(mode_path)
        predict_values = model.predict(x, verbose=0)
        predict_values = np.argmax(predict_values, axis=1)
        test_y = np.argmax(y, axis=1)
        result = classification_report(test_y, predict_values, output_dict=True)
        f1 = f1_score(test_y, predict_values, average='macro')

        fl_dict[Net_Structure] = {}
        fl_dict[Net_Structure]['accuracy'] = result['accuracy']
        fl_dict[Net_Structure]['precision'] = result['macro avg']['precision']
        fl_dict[Net_Structure]['recall'] = result['macro avg']['recall']
        fl_dict[Net_Structure]['f1-score'] = result['macro avg']['f1-score']
        df_f1 = pd.DataFrame.from_dict(fl_dict, 'index').rename_axis('model').reset_index()
        f1_cnt = f1_cnt + 1
        df_f1.to_csv(
            'results/F1_performance_{}.csv'.format(Net_Structure),
            index=False)
        rez_path = "results/total_performance_{}.csv".format(Net_Structure)
        classifaction_report_csv(result, rez_path)

    tmplate_cnt = tmplate_cnt + 1
