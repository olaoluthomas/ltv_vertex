# -*- coding: utf-8 -*-
import logging
import re
import pandas as pd
from lifetimes import ModifiedBetaGeoFitter, GammaGammaFitter
from sklearn.metrics import mean_absolute_error


def snakify(column_name):
    """
    Function to convert pandas column names into snake case.
    CSV's created by SAS have UPPERCASE column names.
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', column_name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def load_data(data):
    """
    Load dataset into a pandas dataframe from csv.
    Args:
        customer_id: key
        frequency_cal: # of unique days of txn.
        recency_cal: # of days between first and last txn.
        T_cal: # of days between first txn and last day of time period.
        monetary_value: avg. product margin (per txn)
    """
    data = pd.read_csv(
        data)  # I need to set up ingestion from cloud storage / BQ
    logging.info('Data loaded to pandas DF.')
    data.columns = [
        snakify(col) if col not in ("T_CAL", "t_cal") else 'T_cal'
        for col in data.columns
    ]
    logging.info('Data columns modified to conform to package rules')
    key = 'customer_id'
    data[key] = data[key].astype("object")
    return data


def mbg_fitter(data, penalty=0.1):
    """
    Fit calibration data to derive Modified BG/NBD parameters.
    """
    mbg = ModifiedBetaGeoFitter(penalizer_coef=penalty)
    mbg.fit(data['frequency_cal'], data['recency_cal'], data['T_cal'])
    logging.info('MBG parameters fit on training data.')
    return mbg


def ggf_fitter(data, penalty=0):
    """
    Fit repeat calibration data to derive Gamma Gamma mixing parameters.
    """
    ggf = GammaGammaFitter(penalizer_coef=penalty)
    ggf.fit(data['frequency_cal'], data['monetary_value'])
    logging.info('Gamma Gamma parameters fit on repeat customer data.')
    return ggf


def save_training_error(repeat, ggf):
    """
    Save error in repeat-customer average profit estimation.
    """
    return mean_absolute_error(
        repeat['monetary_holdout'].mean(),
        ggf.conditional_expected_average_profit(repeat['frequency_cal'],
                                                repeat['monetary_value']).mean,
    )


def save_pickle(mbg, ggf):
    """
    Save Modified BG/NBD & GG parameters to pickle files.
    """
    mbg.save_model('mbg.pkl', save_data=False, save_generate_data_method=False)
    ggf.save_model('ggf.pkl', save_data=False, save_generate_data_method=False)
    logging.info('Model shape and scale parameters saved to file.')


def run_training(train_data):
    data = load_data(data=train_data)
    mbg = mbg_fitter(data=data)
    repeat = data[data['frequency_cal'] > 0]
    repeat.loc[repeat['monetary_value'] <= 0, ['monetary_value']] = 0.0001
    # checking the independence assumption
    p_corr = repeat[['frequency_cal', 'monetary_value'
                     ]].corr()['frequency_cal']['monetary_value']
    if abs(p_corr) < 0.1:
        ggf = ggf_fitter(data=repeat)
        logging.info(
            "Expected conditional average profit: %s, Average future profit: %s"
            % (round(
                ggf.conditional_expected_average_profit(
                    repeat['frequency_cal'], repeat['monetary_value']).mean(),
                2), round(repeat['monetary_holdout'].mean(), 2)))
        absolute_val_error = save_training_error(
            repeat=repeat, ggf=ggf)  # how to spit this out to console
        logging.info("Absolute error: ", round(absolute_val_error, 3))
        save_pickle(mbg=mbg, ggf=ggf)
        # better yet, save error to file or print to console...

    else:
        logging.warning(
            "Pearson correlation co-efficient {} appears to be too high (independence assumption violated...)"
            .format(round(p_corr, 5)))