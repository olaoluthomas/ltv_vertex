# -*- coding: utf-8 -*-
from lifetimes import ModifiedBetaGeoFitter, GammaGammaFitter
from src.trainer import load_data
import logging

output_file = "data/predictions.csv"


def load_params(mbg_, ggf_):
    """
    Load model parameters.
    """
    mbg = ModifiedBetaGeoFitter()
    ggf = GammaGammaFitter()
    mbg.load_model(mbg_)
    ggf.load_model(ggf_)
    logging.info('Model shape and scale parameters loaded.')
    return mbg, ggf


def alive(data, mbg):
    """
    Predict likelihood of customers to be active purchasers
    """
    return mbg.conditional_probability_alive(data['frequency_cal'],
                                             data['recency_cal'],
                                             data['T_cal'])


def ltv_predict(data, mbg, ggf, discount_rate, time):
    """
    Predict dollar value of customers from today up to time t.
    """
    return ggf.customer_lifetime_value(mbg,
                                       data['frequency_cal'],
                                       data['recency_cal'],
                                       data['T_cal'],
                                       data['monetary_value'],
                                       time=time,
                                       discount_rate=discount_rate)


def run_model(input_file, t=12, r=0.00764):
    """
    Initialize model, run predictions, and save output to file.
    """
    # check that pkl files exist.
    mbg, ggf = load_params("mbg.pkl", "ggf.pkl")
    data = load_data(data=input_file)
    logging.info('Data loaded successfully...')
    data['p_alive'] = alive(data, mbg)
    data['prediction'] = round(
        ltv_predict(data, mbg, ggf, time=t, discount_rate=r), 2)
    logging.info('Data scored successfully...')
    data[['customer_id', 'p_alive', 'prediction']].to_csv(
        output_file, index=False,
        encoding='utf-8')  # set up dumping result to BQ/CS or console print.
    logging.info('Predictions saved to csv...')
    