########################################################################################################################
# This is a transition from MATLAB to Python of the FastPLDA toolkit: https://sites.google.com/site/fastplda/
########################################################################################################################


import numpy as np
import collections
from keras.utils import np_utils
import scipy.io
from sklearn.preprocessing import LabelEncoder


def fit_plda_model_simplified(data, labels_cat, numiter, vdim, udim, rcond=1e-6):
    """
    Fits a simplified plda model to a given dataset.
    :param data: data being used for fitting the model
    :param labels_cat: corresponding categorical labels of the data (needs to be boolean!)
    :param numiter: number of iterations used for training the model
    :param vdim: dimension of the first hidden variable
    :param udim: dimension of the second hidden variable
    :return: the trained plda model
    This code has been created with the aid of the fastPLDA toolkit code for MATLAB available at:
    https://sites.google.com/site/fastplda/
    """
    # convert categorical labels to boolean even if label smoothing has been used before
    if not labels_cat.dtype == np.bool:
        labels_cat = (labels_cat == labels_cat.max(axis=1)[:, None]).astype(np.bool)

    # center data in case it has not been gaussianized before
    mu = np.mean(data, axis=0)
    data = (data-mu).transpose()

    # compute moments
    num_classes = labels_cat.shape[1]
    orig_dim = data.shape[0]
    zero_order_moment = data.shape[1]
    first_order_moment = np.zeros((orig_dim, num_classes))
    for k in np.arange(num_classes):
        first_order_moment[:, k] = np.sum(data[:, labels_cat[:, k]], axis=1)
    second_order_moment = np.matmul(data, data.transpose())
    sigma = second_order_moment/zero_order_moment

    # random initialization
    v = np.random.rand(orig_dim, vdim)
    u = np.random.rand(orig_dim, udim)

    # run EM-algorithm
    for k in np.arange(numiter):
        print('Training PLDA model, EM iteration '+str(k+1)+'/'+str(numiter))
        # initialize helpers for em-step
        t = np.zeros((vdim+udim, orig_dim))
        r_yy = np.zeros((vdim, vdim))
        ey = np.zeros((vdim, num_classes))
        y_md = np.zeros((vdim, vdim))

        # E-step
        lamb = np.linalg.inv(sigma)
        q = np.linalg.inv(np.matmul(np.matmul(u.transpose(), lamb), u)+np.identity(udim))
        j = np.matmul(np.matmul(u.transpose(), lamb), v)
        h = v-np.matmul(np.matmul(u, q), j)
        lh = np.matmul(lamb, h)
        vlh = np.matmul(v.transpose(), lh)
        n_old = 0
        for l in np.arange(num_classes):
            n = np.sum(labels_cat[:, l])
            if not n == n_old:
                m = np.linalg.inv(n*vlh+np.identity(vdim))
                n_old = n
            ey[:, l] = np.matmul(m, np.matmul(lh.transpose(), first_order_moment[:, l]))
            eyy = np.matmul(np.reshape(ey[:, l], (vdim, 1)), np.reshape(ey[:, l], (1, vdim)))
            y_md = y_md+(m+eyy)
            r_yy = r_yy+n*(m+eyy)
        y_md = y_md/num_classes
        t_y = np.matmul(ey, first_order_moment.transpose())
        t_x = np.matmul(q, np.matmul(np.matmul(u.transpose(), lamb), second_order_moment)-np.matmul(j, t_y))
        r_yx = np.matmul(np.matmul(np.matmul(t_y, lamb), u)-np.matmul(r_yy, j.transpose()), q)
        w1 = np.matmul(lamb, u)
        w2 = np.matmul(j, t_y)
        r_xx = np.matmul(q, np.matmul(np.matmul(np.matmul(w1.transpose(), second_order_moment), w1)
                                      -np.matmul(w1.transpose(), w2.transpose())-np.matmul(w2, w1)
                                      +np.matmul(np.matmul(j, r_yy), j.transpose()), q))+zero_order_moment*q
        t = np.concatenate((t_y, t_x), axis=0)
        r = np.concatenate((np.concatenate((r_yy, r_yx), axis=1),
                            np.concatenate((r_yx.transpose(), r_xx), axis=1)), axis=0)

        # M-step
        vu = np.linalg.lstsq(r, t, rcond=rcond)[0].transpose()
        v = vu[:, 0:vdim]
        u = vu[:, vdim:]
        sigma = (second_order_moment-np.matmul(vu, t))/zero_order_moment

        # minimum-divergence step
        g = np.linalg.lstsq(r_yy, r_yx, rcond=rcond)[0].transpose()
        x_md = (r_xx-np.matmul(g, r_yx))/zero_order_moment
        u = np.matmul(u, np.linalg.cholesky(x_md+np.eye(x_md.shape[0])*rcond))
        v = np.matmul(v, np.linalg.cholesky(y_md+np.eye(y_md.shape[0])*rcond))+np.matmul(u, g)

    # return trained model
    Model = collections.namedtuple('Model', ['v', 'u', 'mu', 'sigma'])
    model = Model(v=v, u=u, mu=mu, sigma=sigma)
    return model


def fit_plda_model_two_cov(data, labels_cat, numiter):
    """
    Fits a two-covariance plda model to a given dataset.
    :param data: data being used for fitting the model
    :param labels_cat: corresponding categorical labels of the data (needs to be boolean!)
    :param numiter: number of iterations used for training the model
    :return: the trained plda model
    This code has been created with the aid of the fastPLDA toolkit code for MATLAB available at:
    https://sites.google.com/site/fastplda/
    """
    # convert categorical labels to boolean even if label smoothing has been used before
    if not labels_cat.dtype == np.bool:
        labels_cat = (labels_cat == labels_cat.max(axis=1)[:, None]).astype(np.bool)

    # center data in case it has not been gaussianized before
    mu = np.mean(data, axis=0)
    data = (data-mu).transpose()

    # compute moments
    num_classes = labels_cat.shape[1]
    orig_dim = data.shape[0]
    zero_order_moment = data.shape[1]
    first_order_moment = np.zeros((orig_dim, num_classes))
    for k in np.arange(num_classes):
        first_order_moment[:, k] = np.sum(data[:, labels_cat[:, k]], axis=1)
    second_order_moment = np.matmul(data, data.transpose())

    # initialize
    invb = second_order_moment/zero_order_moment
    invw = second_order_moment/zero_order_moment

    # run EM-algorithm
    for k in np.arange(numiter):
        print('Training PLDA model, EM iteration '+str(k+1)+'/'+str(numiter))

        # E-step
        b = np.linalg.inv(invb)
        w = np.linalg.inv(invw)
        t = np.zeros((orig_dim, orig_dim))
        r = np.zeros((orig_dim, orig_dim))
        y = np.zeros((orig_dim, ))
        bmu = np.matmul(b, mu)
        n_old = 0
        for l in np.arange(num_classes):
            n = np.sum(labels_cat[:, l])
            if not n == n_old:
                invl_i = np.linalg.inv(b + n * w)
                n_old = n
            ey_i = np.matmul(invl_i, bmu + np.matmul(w, first_order_moment[:, l]))
            t = t + np.outer(ey_i, first_order_moment[:, l])
            r = r + n*(invl_i + np.outer(ey_i, ey_i))
            y = y + n*ey_i

        # M-step
        mu = y / zero_order_moment
        invb = (r - np.outer(mu, y.transpose()) - np.outer(y, mu))/zero_order_moment + np.outer(mu, mu)
        invw = (second_order_moment - t - t.transpose() + r)/zero_order_moment

    # return trained model
    Model = collections.namedtuple('Model', ['invb', 'invw', 'mu'])
    model = Model(invb=invb, invw=invw, mu=mu)
    return model


def get_plda_scores_simplified(model, data1, data2):
    """
    Computes the log-likelihood ratios between two datasets for a plda model.
    :param model: plda model used for the evaluation
    :param data1: first dataset
    :param data2: second dataset
    :return: log-likelihood ratios as scores
    This code has been created with the aid of the fastPLDA toolkit code for MATLAB available at:
    https://sites.google.com/site/fastplda/
    """
    # center data in case it has not been gaussianized before
    data1 = data1-model.mu
    data2 = data2-model.mu

    # compute log-likelihood ratios
    sigma_wc = np.matmul(model.u, model.u.transpose())+model.sigma
    sigma_ac = np.matmul(model.v, model.v.transpose())
    sigma_tot = sigma_wc+sigma_ac
    h1 = -np.linalg.inv(sigma_wc+2*sigma_ac)
    h2 = np.linalg.inv(sigma_wc)
    lamb_tot = h1+h2
    gamma = h1-h2+2*np.linalg.inv(sigma_tot)
    gamma11 = np.sum(np.multiply(np.matmul(data1, gamma), data1), axis=1)
    gamma22 = np.sum(np.multiply(np.matmul(data2, gamma), data2), axis=1)
    scores = 2*np.matmul(np.matmul(data1, lamb_tot), data2.transpose())+gamma11[:, np.newaxis]+gamma22
    return scores


def get_plda_scores_two_cov(model, data1, data2):
    """
    Computes the log-likelihood ratios between two datasets for a plda model.
    :param model: plda model used for the evaluation
    :param data1: first dataset
    :param data2: second dataset
    :return: log-likelihood ratios as scores
    This code has been created with the aid of the fastPLDA toolkit code for MATLAB available at:
    https://sites.google.com/site/fastplda/
    """
    # center data in case it has not been gaussianized before
    data1 = data1-model.mu
    data2 = data2-model.mu

    # compute log-likelihood ratios
    sigma_wc = model.invw
    sigma_ac = model.invb
    sigma_tot = sigma_wc+sigma_ac
    h1 = -np.linalg.inv(sigma_wc+2*sigma_ac)
    h2 = np.linalg.inv(sigma_wc)
    lamb_tot = h1+h2
    gamma = h1-h2+2*np.linalg.inv(sigma_tot)
    gamma11 = np.sum(np.multiply(np.matmul(data1, gamma), data1), axis=1)
    gamma22 = np.sum(np.multiply(np.matmul(data2, gamma), data2), axis=1)
    scores = 2*np.matmul(np.matmul(data1, lamb_tot), data2.transpose())+gamma11[:, np.newaxis]+gamma22
    return scores


def gaussianize(data, m=None, w=None):
    """
    CAUTION: Does not seem to work correct
    Projects data to the unitsphere
    :param data: data to be projected
    :param m: optional mean value, if not set it will be computed from the given data
    :param w: optional whitening transformation, if not set it will be computed from the given data
    :return: the projected data
    """
    if m is None:
        m = np.mean(data, axis=0)
    if w is None:
        s = np.cov(data, rowvar=False)
        d, q = np.linalg.eig(s)
        w = np.matmul(np.diag(1/np.sqrt(d)), q.transpose())
    data = np.matmul((data-m), w.transpose())
    return data, m, w


if __name__ == '__main__':
    # params
    numiter = 10
    vdim = 120
    udim = 120
    plda_type = 'two_cov'  # 'two_cov' or 'simplified'

    # load data
    print('Loading data')
    mat = scipy.io.loadmat('demo_data.mat')
    train_data = mat['train_data']
    train_labels = mat['train_labels']
    enrol_data = mat['enrol_data']
    enrol_labels = mat['enrol_labels']
    test_data = mat['test_data']
    test_labels = mat['test_labels']

    # encode labels
    le = LabelEncoder()
    train_labels_enc = le.fit_transform(train_labels.ravel())
    num_classes = len(le.classes_)
    train_labels_cat = np_utils.to_categorical(train_labels_enc, num_classes=num_classes)

    # average enrolment data
    enrol_data_avr = np.zeros((np.unique(enrol_labels).shape[0], enrol_data.shape[1]))
    for k, l in enumerate(np.unique(enrol_labels)):
        enrol_data_avr[k, :] = np.mean(enrol_data[(enrol_labels == l)[:, 0], :], axis=0)

    # gaussianize
    print('Gaussianizing data')
    train_data, m, w = gaussianize(train_data)
    enrol_data, _, _ = gaussianize(test_data, m, w)
    test_data, _, _ = gaussianize(test_data, m, w)

    if plda_type == 'two_simplified':
        # train plda model
        print('Training plda model')
        plda_model = fit_plda_model_simplified(train_data, train_labels_cat, numiter=numiter, vdim=vdim, udim=udim)

        # compute scores for test data
        print('Computing test scores')
        scores = get_plda_scores_simplified(plda_model, enrol_data_avr, test_data)
    elif plda_type == 'two_cov':
        # train plda model
        print('Training plda model')
        plda_model = fit_plda_model_two_cov(train_data, train_labels_cat, numiter=numiter)

        # compute scores for test data
        print('Computing test scores')
        scores = get_plda_scores_two_cov(plda_model, enrol_data_avr, test_data)
    print(scores)
