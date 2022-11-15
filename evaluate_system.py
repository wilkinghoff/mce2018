import numpy as np
from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
import plda
import keras
import os


def adaptive_snorm(scores, scores_enr, scores_test, n_cohort_enr=200, n_cohort_test=200):
    scores_enr = -np.sort(-scores_enr, axis=1)[:, :n_cohort_enr]
    scores_test = -np.sort(-scores_test, axis=1)[:, :n_cohort_test]
    mean_enr = np.tile(np.expand_dims(np.mean(scores_enr, axis=1), axis=1), (1, scores.shape[1]))
    mean_test = np.tile(np.expand_dims(np.mean(scores_test, axis=1), axis=0), (scores.shape[0], 1))
    std_enr = np.tile(np.expand_dims(np.std(scores_enr, axis=1), axis=1), (1, scores.shape[1]))
    std_test = np.tile(np.expand_dims(np.std(scores_test, axis=1), axis=0), (scores.shape[0], 1))
    return 0.5*((scores-mean_enr)/std_enr+(scores-mean_test)/std_test)


def load_ivector(filename):
    utt = np.loadtxt(filename, dtype=np.str, delimiter=',', skiprows=1, usecols=[0])
    ivector = np.loadtxt(filename, dtype=np.float, delimiter=',', skiprows=1, usecols=range(1, 601))
    spk_id = []
    for iter in range(len(utt)):
        spk_id = np.append(spk_id, utt[iter].split('_')[0])
    return spk_id, utt, ivector


def length_norm(mat):
    """
    length normalization (l2 norm)
    input: mat = [utterances X vector dimension] ex) (float) 8631 X 600
    """
    norm_mat = []
    for line in mat:
        temp = line/np.math.sqrt(sum(np.power(line, 2)))
        norm_mat.append(temp)
    norm_mat = np.array(norm_mat)
    return norm_mat


def make_spkvec(mat, spk_label):
    """
    calculating speaker mean vector
    input: mat = [utterances X vector dimension] ex) (float) 8631 X 600
           spk_label = string vector ex) ['abce','cdgd']

        for iter in range(len(spk_label)):
            spk_label[iter] = spk_label[iter].split('_')[0]
    """

    spk_label, spk_index = np.unique(spk_label, return_inverse=True)
    spk_mean = []
    mat = np.array(mat)

    # calculating speaker mean i-vector
    for i, spk in enumerate(spk_label):
        spk_mean.append(np.mean(mat[np.nonzero(spk_index == i)], axis=0))
    spk_mean = length_norm(spk_mean)
    return spk_mean, spk_label


def calculate_EER(trials, scores):
    # calculating EER of Top-S detector
    # input: trials = boolean(or int) vector, 1: positive(blacklist) 0: negative(background)
    #        scores = float vector

    # Calculating EER
    fpr, tpr, threshold = roc_curve(trials, scores, pos_label=1)
    fnr = 1 - tpr
    EER_threshold = threshold[np.argmin(abs(fnr - fpr))]

    # print EER_threshold
    EER_fpr = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    EER_fnr = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    EER = 0.5 * (EER_fpr + EER_fnr)

    print("Top S detector EER is %0.2f%%" % (EER * 100))
    return EER


def get_trials_label_with_confusion(identified_label, groundtruth_label, dict4spk, is_trial):
    # determine if the test utterance would make confusion error
    # input: identified_label = string vector, identified result of test utterance among multi-target from the detection system
    #        groundtruth_label = string vector, ground truth speaker labels of test utterances
    #        dict4spk = dictionary, convert label to target set, ex) train2dev convert train id to dev id

    trials = np.zeros(len(identified_label))
    for iter in range(0, len(groundtruth_label)):
        enroll = identified_label[iter].split('_')[0]
        test = groundtruth_label[iter].split('_')[0]
        if is_trial[iter]:
            if enroll == dict4spk[test]:
                trials[iter] = 1  # for Target trial (blacklist speaker)
            else:
                trials[iter] = -1  # for Target trial (blacklist speaker), but fail on blacklist classifier

        else:
            trials[iter] = 0  # for non-target (non-blacklist speaker)
    return trials


def calculate_EER_with_confusion(scores, trials):
    # calculating EER of Top-1 detector
    # input: trials = boolean(or int) vector, 1: postive(blacklist) 0: negative(background) -1: confusion(blacklist)
    #        scores = float vector

    # exclude confusion error (trials==-1)
    scores_wo_confusion = scores[np.nonzero(trials != -1)[0]]
    trials_wo_confusion = trials[np.nonzero(trials != -1)[0]]

    # dev_trials contain labels of target. (target=1, non-target=0)
    fpr, tpr, threshold = roc_curve(trials_wo_confusion, scores_wo_confusion, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    EER_threshold = threshold[np.argmin(abs(fnr - fpr))]

    # EER withouth confusion error
    EER = fpr[np.argmin(np.absolute((fnr - fpr)))]

    # Add confusion error to false negative rate(Miss rate)
    total_negative = len(np.nonzero(np.array(trials_wo_confusion) == 0)[0])
    total_positive = len(np.nonzero(np.array(trials_wo_confusion) == 1)[0])
    fp = fpr * np.float(total_negative)
    fn = fnr * np.float(total_positive)
    fn += len(np.nonzero(trials == -1)[0])
    total_positive += len(np.nonzero(trials == -1)[0])
    fpr = fp / total_negative
    fnr = fn / total_positive

    # EER with confusion Error
    EER_threshold = threshold[np.argmin(abs(fnr - fpr))]
    EER_fpr = fpr[np.argmin(np.absolute((fnr - fpr)))]
    EER_fnr = fnr[np.argmin(np.absolute((fnr - fpr)))]
    EER = 0.5 * (EER_fpr + EER_fnr)

    print("Top 1 detector EER is %0.2f%% (Total confusion error is %d)" % ((EER * 100), len(np.nonzero(trials == -1)[0])))
    return EER


if __name__ == '__main__':
    # making dictionary to find blacklist pair between train and test dataset
    bl_match = np.loadtxt('data/bl_matching.csv', dtype=np.str)
    dev2train = {}
    dev2id = {}
    train2dev = {}
    train2id = {}
    test2train = {}
    train2test = {}
    for iter, line in enumerate(bl_match):
        line_s = line.split(',')
        dev2train[line_s[1].split('_')[-1]] = line_s[3].split('_')[-1]
        dev2id[line_s[1].split('_')[-1]] = line_s[0].split('_')[-1]
        train2dev[line_s[3].split('_')[-1]] = line_s[1].split('_')[-1]
        train2id[line_s[3].split('_')[-1]] = line_s[0].split('_')[-1]
        test2train[line_s[2].split('_')[-1]] = line_s[3].split('_')[-1]
        train2test[line_s[3].split('_')[-1]] = line_s[2].split('_')[-1]

    # Loading i-vector
    print('Loading i-vectors')
    trn_bl_id, trn_bl_utt, trn_bl_ivector = load_ivector('data/trn_blacklist.csv')
    trn_bg_id, trn_bg_utt, trn_bg_ivector = load_ivector('data/trn_background.csv')
    dev_bl_id, dev_bl_utt, dev_bl_ivector = load_ivector('data/dev_blacklist.csv')
    dev_bg_id, dev_bg_utt, dev_bg_ivector = load_ivector('data/dev_background.csv')

    # load dev and test set information
    filename = 'data/tst_evaluation_keys.csv'
    tst_info = np.loadtxt(filename, dtype=np.str, delimiter=',', skiprows=1, usecols=range(0, 3))
    tst_trials = []
    tst_trials_label = []
    tst_ground_truth = []
    for iter in range(len(tst_info)):
        tst_trials_label.extend([tst_info[iter, 0]])
        if tst_info[iter, 1] == 'background':
            tst_trials = np.append(tst_trials, 0)
        else:
            tst_trials = np.append(tst_trials, 1)
    dev_trials = np.append(np.ones([len(dev_bl_id), 1]), np.zeros([len(dev_bg_id), 1]))

    # convert labels of development set
    dev_trials_label = np.append(dev_bl_id, dev_bg_id)
    dev_bl_id = [dev2train[dev_id] for dev_id in dev_bl_id]
    dev_ivector = np.append(dev_bl_ivector, dev_bg_ivector, axis=0)
    trn_ivector = np.append(trn_bl_ivector, trn_bg_ivector, axis=0)

    # encode labels
    le = LabelEncoder()
    trn_id_enc = le.fit_transform(trn_bl_id)
    num_bl_spk = len(np.unique(trn_id_enc))
    trn_id_cat = np_utils.to_categorical(trn_id_enc, num_classes=num_bl_spk)
    dev_id_cat = np_utils.to_categorical(le.transform(dev_bl_id), num_classes=num_bl_spk)
    le_bg = LabelEncoder()
    trn_bg_id_enc = le_bg.fit_transform(trn_bg_id)+num_bl_spk
    num_bg_spk = len(np.unique(trn_bg_id_enc))

    # get reconstruction targets
    spk_mean, spk_mean_label = make_spkvec(trn_bl_ivector, trn_bl_id)
    spk_mean_ln = length_norm(spk_mean)
    spk_mean_trn = np.zeros(trn_bl_ivector.shape)
    for i, l in enumerate(trn_bl_id):
        spk_mean_trn[i] = spk_mean[spk_mean_label == l]
    spk_mean_trn_ln = length_norm(spk_mean_trn)

    # estimate transformation for linear alignment
    print('estimating transformation for linear alignment')
    epochs = 400
    batch_size = 128
    inputs = keras.layers.Input(shape=(600,))
    x = keras.layers.Dense(600, activation='linear')(inputs)
    model = keras.models.Model(inputs=inputs, outputs=x)
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.adam(lr=0.0001))
    # fit model
    weight_path = './trn_to_mean.h5'
    callbacks = [
        keras.callbacks.ModelCheckpoint(weight_path, monitor='loss', mode='min', verbose=1, save_best_only=True)
    ]
    if not os.path.isfile(weight_path):
        # train model
        model.fit(trn_bl_ivector, spk_mean_trn, batch_size=batch_size, verbose=2,
                  epochs=epochs, shuffle=True, callbacks=callbacks)
        model.save_weights(weight_path)
    model.load_weights(weight_path)

    # preprocessing
    print('preprocessing i-vectors')
    trn_ivector_la = model.predict(trn_ivector, batch_size=trn_ivector.shape[0])
    trn_bl_ivector_la = trn_ivector_la[:trn_bl_ivector.shape[0]]
    trn_bg_ivector_la = trn_ivector_la[trn_bl_ivector.shape[0]:]
    trn_ivector_ln_la = length_norm(trn_ivector_la)
    trn_bl_ivector_ln_la = length_norm(trn_bl_ivector_la)
    trn_bg_ivector_ln_la = length_norm(trn_bg_ivector_la)
    dev_ivector_la = model.predict(dev_ivector, batch_size=dev_ivector.shape[0])
    dev_bl_ivector_la = dev_ivector_la[:dev_bl_ivector.shape[0]]
    dev_bg_ivector_la = dev_ivector_la[dev_bl_ivector.shape[0]:]
    dev_ivector_ln_la = length_norm(dev_ivector_la)
    dev_bl_ivector_ln_la = length_norm(dev_bl_ivector_la)
    dev_bg_ivector_ln_la = length_norm(dev_bg_ivector_la)

    # Evaluating Cosine Similarity with M-Norm
    print('evaluating cosine similarity with M-Norm')
    spk_mean_la, spk_mean_label = make_spkvec(trn_bl_ivector_la, trn_bl_id)
    spk_mean_ln_la = length_norm(spk_mean_la)
    spk_mean_bg_la, spk_mean_bg_label = make_spkvec(trn_bg_ivector_la, trn_bg_id)
    spk_mean_bg_ln_la = length_norm(spk_mean_bg_la)
    dev_scores_la = spk_mean_ln_la.dot(dev_ivector_ln_la.transpose())
    trn_bl_ivector_ln = length_norm(trn_bl_ivector)  # using trn_bl_ivector_ln_la would be too optimistic
    blscores = spk_mean_ln.dot(trn_bl_ivector_ln.transpose())
    mnorm_mu_bl = np.mean(blscores, axis=1)
    for iter in range(np.shape(dev_scores_la)[1]):
        dev_scores_la[:, iter] = (dev_scores_la[:, iter]-mnorm_mu_bl)

    # apply LDA for PLDA
    print('applying LDA')
    clf_deep = LinearDiscriminantAnalysis(n_components=600)
    clf_deep.fit(trn_bl_ivector_ln_la, trn_id_enc)
    trn_bl_ivector_lda = clf_deep.transform(trn_bl_ivector_ln_la)
    trn_bg_ivector_lda = clf_deep.transform(trn_bg_ivector_ln_la)
    trn_ivector_lda = clf_deep.transform(trn_ivector_ln_la)
    dev_ivector_lda = clf_deep.transform(dev_ivector_ln_la)
    trn_bl_ivector_lda = length_norm(trn_bl_ivector_lda)
    trn_bg_ivector_lda = length_norm(trn_bg_ivector_lda)
    trn_ivector_lda = length_norm(trn_ivector_lda)
    dev_ivector_lda = length_norm(dev_ivector_lda)
    spk_mean_lda, spk_mean_label = make_spkvec(trn_bl_ivector_lda, trn_bl_id)
    spk_mean_bg_lda, spk_mean_bg_label = make_spkvec(trn_bg_ivector_lda, trn_bg_id)

    # evalute lda-plda
    print('training LDA-PLDA model')
    numiter = 20
    lda_plda_model = plda.fit_plda_model_two_cov(trn_bl_ivector_lda, trn_id_cat, numiter=numiter)
    dev_scores_lda_plda = plda.get_plda_scores_two_cov(lda_plda_model, spk_mean_lda, dev_ivector_lda)
    dev_ensemble_scores_lda_plda = plda.get_plda_scores_two_cov(lda_plda_model, trn_bl_ivector_lda, dev_ivector_lda)

    # apply as-norm
    print('applying adaptive s-norm')
    dev_scores_cohort_enr_lda_plda = plda.get_plda_scores_two_cov(lda_plda_model, spk_mean_lda, trn_bl_ivector_lda)
    dev_scores_cohort_test_lda_plda = plda.get_plda_scores_two_cov(lda_plda_model, dev_ivector_lda, trn_bl_ivector_lda)
    dev_scores_lda_plda = adaptive_snorm(dev_scores_lda_plda, dev_scores_cohort_enr_lda_plda,
                                         dev_scores_cohort_test_lda_plda, n_cohort_enr=700, n_cohort_test=9000)

    # evalute plda
    print('training PLDA model')
    numiter = 20
    plda_model = plda.fit_plda_model_two_cov(np.concatenate([trn_bl_ivector_ln_la, trn_bg_ivector_ln_la], axis=0),
                                             np_utils.to_categorical(np.concatenate([trn_id_enc, trn_bg_id_enc], axis=0), num_classes=num_bl_spk+num_bg_spk), numiter=numiter)
    dev_scores_plda = plda.get_plda_scores_two_cov(plda_model, spk_mean_ln_la, dev_ivector_ln_la)
    dev_ensemble_scores_plda = plda.get_plda_scores_two_cov(plda_model, np.concatenate([trn_bl_ivector_ln_la, trn_bg_ivector_ln_la], axis=0), dev_ivector_ln_la)

    # apply s-norm
    print('applying adaptive s-norm')
    dev_scores_cohort_enr_plda = plda.get_plda_scores_two_cov(plda_model, spk_mean_ln_la, trn_bg_ivector_ln_la)
    dev_scores_cohort_test_plda = plda.get_plda_scores_two_cov(plda_model, dev_ivector_ln_la, trn_bg_ivector_ln_la)
    dev_scores_plda = adaptive_snorm(dev_scores_plda, dev_scores_cohort_enr_plda,
                                     dev_scores_cohort_test_plda, n_cohort_enr=2800, n_cohort_test=600)

    # closed-set identification
    dev_identified_label = spk_mean_label[(np.argmax(dev_scores_lda_plda, axis=0))]
    dev_trials_confusion = get_trials_label_with_confusion(dev_identified_label, dev_trials_label, dev2train, dev_trials)

    print('results on development set:')
    #take maximum
    dev_scores_la = np.max(dev_scores_la, axis=0)
    dev_scores_plda = np.max(dev_scores_plda, axis=0)
    dev_ensemble_scores_plda = np.max(dev_ensemble_scores_plda, axis=0)
    dev_ensemble_scores_lda_plda = np.max(dev_ensemble_scores_lda_plda, axis=0)

    # output top-S EER
    print('top-S EER obtained with cosine similarity:')
    dev_EER_la = calculate_EER(dev_trials, dev_scores_la)
    print('top-S EER obtained with PLDA:')
    dev_EER_plda = calculate_EER(dev_trials, dev_scores_plda)
    print('top-S EER obtained with ensemble:')
    clf_ensemble = LogisticRegression(class_weight='balanced', solver='saga', penalty='l2', max_iter=10000, n_jobs=-1).fit(np.vstack([dev_scores_plda,
                                                                                                                                      dev_ensemble_scores_plda,
                                                                                                                                      dev_ensemble_scores_lda_plda]).transpose(), dev_trials)
    dev_scores_fused = clf_ensemble.decision_function(np.vstack([dev_scores_plda,
                                                                 dev_ensemble_scores_plda,
                                                                 dev_ensemble_scores_lda_plda]).transpose())
    dev_EER_fused = calculate_EER(dev_trials, dev_scores_fused)

    # top-1 detector EER
    print('top-1 EER obtained with single model:')
    dev_EER_confusion = calculate_EER_with_confusion(dev_scores_plda, dev_trials_confusion)
    print('top-1 EER obtained with ensemble:')
    dev_EER_confusion = calculate_EER_with_confusion(dev_scores_fused, dev_trials_confusion)

    ####################################################################################################################
    # Repeat for test set
    ####################################################################################################################
    # Loading i-vector
    print('loading i-vectors (again)')
    _, _, trn_bl_ivector = load_ivector('data/trn_blacklist.csv')
    _, _, trn_bg_ivector = load_ivector('data/trn_background.csv')
    _, _, dev_bl_ivector = load_ivector('data/dev_blacklist.csv')
    _, _, dev_bg_ivector = load_ivector('data/dev_background.csv')
    tst_id, tst_utt, tst_ivector = load_ivector('data/tst_evaluation.csv')

    # renaming for submission
    trn_bl_id = np.concatenate([trn_bl_id, dev_bl_id], axis=0)
    trn_bg_id = np.concatenate([trn_bg_id, dev_bg_id], axis=0)
    trn_bl_ivector = np.concatenate([trn_bl_ivector, dev_bl_ivector], axis=0)
    trn_bg_ivector = np.concatenate([trn_bg_ivector, dev_bg_ivector], axis=0)
    trn_ivector = np.append(trn_bl_ivector, trn_bg_ivector, axis=0)

    # encode labels
    le = LabelEncoder()
    trn_id_enc = le.fit_transform(trn_bl_id)
    num_bl_spk = len(np.unique(trn_id_enc))
    trn_id_cat = np_utils.to_categorical(trn_id_enc, num_classes=num_bl_spk)
    dev_id_cat = np_utils.to_categorical(le.transform(dev_bl_id), num_classes=num_bl_spk)
    le_bg = LabelEncoder()
    trn_bg_id_enc = le_bg.fit_transform(trn_bg_id)+num_bl_spk
    num_bg_spk = len(np.unique(trn_bg_id_enc))

    # get reconstruction targets
    spk_mean, spk_mean_label = make_spkvec(trn_bl_ivector, trn_bl_id)
    spk_mean_ln = length_norm(spk_mean)
    spk_mean_trn = np.zeros(trn_bl_ivector.shape)
    for i, l in enumerate(trn_bl_id):
        spk_mean_trn[i] = spk_mean[spk_mean_label == l]
    spk_mean_trn_ln = length_norm(spk_mean_trn)

    # estimate transformation for linear alignment
    print('estimating transformation for linear alignment')
    epochs = 400
    batch_size = 128
    inputs = keras.layers.Input(shape=(600,))
    x = keras.layers.Dense(600, activation='linear')(inputs)
    model = keras.models.Model(inputs=inputs, outputs=x)
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.adam(lr=0.0001))
    # fit model
    weight_path = './trn_and_dev_to_mean.h5'
    callbacks = [
        keras.callbacks.ModelCheckpoint(weight_path, monitor='loss', mode='min', verbose=1, save_best_only=True)
    ]
    if not os.path.isfile(weight_path):
        # train model
        model.fit(trn_bl_ivector, spk_mean_trn, batch_size=batch_size, verbose=2,
                  epochs=epochs, shuffle=True, callbacks=callbacks)
        model.save_weights(weight_path)
    model.load_weights(weight_path)

    # preprocessing
    print('preprocessing i-vectors')
    trn_ivector_la = model.predict(trn_ivector, batch_size=trn_ivector.shape[0])
    trn_bl_ivector_la = trn_ivector_la[:trn_bl_ivector.shape[0]]
    trn_bg_ivector_la = trn_ivector_la[trn_bl_ivector.shape[0]:]
    trn_ivector_ln_la = length_norm(trn_ivector_la)
    trn_bl_ivector_ln_la = trn_ivector_ln_la[:trn_bl_ivector.shape[0]]
    trn_bg_ivector_ln_la = trn_ivector_ln_la[trn_bl_ivector.shape[0]:]
    tst_ivector_la = model.predict(tst_ivector, batch_size=tst_ivector.shape[0])
    tst_ivector_ln_la = length_norm(tst_ivector_la)

    # Evaluating Cosine Similarity with M-Norm
    print('evaluating cosine similarity with M-Norm')
    spk_mean_la, spk_mean_label = make_spkvec(trn_bl_ivector_la, trn_bl_id)
    spk_mean_ln_la = length_norm(spk_mean_la)
    spk_mean_bg_la, spk_mean_bg_label = make_spkvec(trn_bg_ivector_la, trn_bg_id)
    spk_mean_bg_ln_la = length_norm(spk_mean_bg_la)
    scores_la = spk_mean_ln_la.dot(tst_ivector_ln_la.transpose())
    trn_bl_ivector_ln = length_norm(trn_bl_ivector)  # using trn_bl_ivector_ln_la would be too optimistic
    blscores = spk_mean_ln.dot(trn_bl_ivector_ln.transpose())
    mnorm_mu_bl = np.mean(blscores, axis=1)
    for iter in range(np.shape(scores_la)[1]):
        scores_la[:, iter] = (scores_la[:, iter]-mnorm_mu_bl)

    # apply LDA for PLDA
    print('applying Linear Discriminant Analysis')
    clf_lda = LinearDiscriminantAnalysis(n_components=600)
    clf_lda.fit(trn_bl_ivector_ln_la, trn_id_enc)
    trn_bl_ivector_lda = clf_lda.transform(trn_bl_ivector_ln_la)
    trn_bg_ivector_lda = clf_lda.transform(trn_bg_ivector_ln_la)
    tst_ivector_lda = clf_lda.transform(tst_ivector_ln_la)
    trn_bl_ivector_lda = length_norm(trn_bl_ivector_lda)
    trn_bg_ivector_lda = length_norm(trn_bg_ivector_lda)
    tst_ivector_lda = length_norm(tst_ivector_lda)
    spk_mean_lda, spk_mean_label = make_spkvec(trn_bl_ivector_lda, trn_bl_id)
    spk_mean_bg_lda, spk_mean_bg_label = make_spkvec(trn_bg_ivector_lda, trn_bg_id)

    # evalute lda-plda
    print('training LDA-PLDA model')
    numiter = 20
    lda_plda_model = plda.fit_plda_model_two_cov(trn_bl_ivector_lda, trn_id_cat, numiter=numiter)
    scores_lda_plda = plda.get_plda_scores_two_cov(lda_plda_model, spk_mean_lda, tst_ivector_lda)
    ensemble_scores_lda_plda = plda.get_plda_scores_two_cov(lda_plda_model, trn_bl_ivector_lda, tst_ivector_lda)

    # apply as-norm
    print('applying adaptive s-norm')
    scores_cohort_enr_lda_plda = plda.get_plda_scores_two_cov(lda_plda_model, spk_mean_lda, trn_bl_ivector_lda)
    scores_cohort_test_lda_plda = plda.get_plda_scores_two_cov(lda_plda_model, tst_ivector_lda, trn_bl_ivector_lda)
    scores_lda_plda = adaptive_snorm(scores_lda_plda, scores_cohort_enr_lda_plda,
                                     scores_cohort_test_lda_plda, n_cohort_enr=700, n_cohort_test=9000)

    # evalute plda
    print('training PLDA model')
    numiter = 20
    plda_model = plda.fit_plda_model_two_cov(np.concatenate([trn_bl_ivector_ln_la, trn_bg_ivector_ln_la], axis=0),
                                             np_utils.to_categorical(np.concatenate([trn_id_enc, trn_bg_id_enc], axis=0), num_classes=num_bl_spk+num_bg_spk), numiter=numiter)
    scores_plda = plda.get_plda_scores_two_cov(plda_model, spk_mean_ln_la, tst_ivector_ln_la)
    ensemble_scores_plda = plda.get_plda_scores_two_cov(plda_model, np.concatenate([trn_bl_ivector_ln_la, trn_bg_ivector_ln_la], axis=0), tst_ivector_ln_la)

    # apply s-norm
    print('applying adaptive s-norm')
    scores_cohort_enr_plda = plda.get_plda_scores_two_cov(plda_model, spk_mean_ln_la, trn_bg_ivector_ln_la)
    scores_cohort_test_plda = plda.get_plda_scores_two_cov(plda_model, tst_ivector_ln_la, trn_bg_ivector_ln_la)
    scores_plda = adaptive_snorm(scores_plda, scores_cohort_enr_plda,
                                 scores_cohort_test_plda, n_cohort_enr=2800, n_cohort_test=600)

    # closed-set identification
    tst_identified_label = spk_mean_label[(np.argmax(scores_lda_plda, axis=0))]
    tst_trials_confusion = get_trials_label_with_confusion(tst_identified_label, tst_trials_label, test2train,
                                                           tst_trials)

    #take maximum
    scores_la = np.max(scores_la, axis=0)
    scores_plda = np.max(scores_plda, axis=0)
    ensemble_scores_plda = np.max(ensemble_scores_plda, axis=0)
    ensemble_scores_lda_plda = np.max(ensemble_scores_lda_plda, axis=0)

    # output top-S EER
    print('top-S EER obtained with cosine similarity:')
    tst_EER_la = calculate_EER(tst_trials, scores_la)
    print('top-S EER obtained with PLDA:')
    tst_EER_plda = calculate_EER(tst_trials, scores_plda)
    print('top-S EER obtained with ensemble:')
    scores_fused = clf_ensemble.decision_function(np.vstack([scores_plda, ensemble_scores_plda,
                                                             ensemble_scores_lda_plda]).transpose())
    tst_EER_fused = calculate_EER(tst_trials, scores_fused)

    # top-1 detector EER
    print('top-1 EER obtained with single model:')
    tst_EER_confusion = calculate_EER_with_confusion(scores_plda, tst_trials_confusion)
    print('top-1 EER obtained with ensemble:')
    tst_EER_confusion = calculate_EER_with_confusion(scores_fused, tst_trials_confusion)

    ####################################################################################################################
    # Generate Submission File
    ####################################################################################################################
    filename = 'Fraunhofer FKIE_fixed_primary.csv'
    # filename = 'Fraunhofer FKIE_fixed_contrastive1.csv'
    with open(filename, "w") as text_file:
        for iter, score in enumerate(scores_fused):
            id_in_trainset = tst_identified_label[iter].split('_')[0]
            input_file = tst_utt[iter]
            text_file.write('%s,%s,%s\n' % (input_file, score, train2id[id_in_trainset]))


