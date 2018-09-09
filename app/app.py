from flask import Flask, render_template, request
import requests
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import StandardScaler
pd.options.display.max_columns=25


def cos_sim_recommendations(new_data, df, index_name, n=1):
    cs = cosine_similarity(new_data, df)
    rec_index = np.argsort(cs)[0][-n-1:][::-1][1:]
    recommendations = []
    for rec in rec_index:
        recommendations.append(index_name[rec])
    return recommendations


app = Flask(__name__, static_url_path='/static')

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/recommender', methods=['GET', 'POST'])
def recommender():
    return render_template('recommender.html')

@app.route('/recommendations', methods=['GET', 'POST'])
def recommendations():
    vendors = pd.read_csv('data/florists_updated.csv')

    vendors = vendors[['#', 'Company Name', 'Total price', 'Location', 'Service', 'Ceremony decor', 'Reception decor', 'Handhelds (bouquets and boutonnieres)', 'Approximate size of wedding (if known)']]

    vendor_cols = ['id', 'company_name', 'total_price', 'location', 'service', 'ceremony_decor', 'reception_decor', 'handhelds', 'size_of_wedding']
    vendors.columns = vendor_cols

    vendors = pd.get_dummies(vendors, columns=['location', 'service'])

    vendors['ceremony_decor'].fillna(0, inplace=True)
    vendors['reception_decor'].fillna(0, inplace=True)
    vendors['handhelds'].fillna(0, inplace=True)

    vendors['ceremony_decor'].replace('Ceremony decor', 1, inplace=True)
    vendors['reception_decor'].replace('Reception decor', 1, inplace=True)
    vendors['handhelds'].replace('Handhelds (bouquets and boutonnieres)', 1, inplace=True)

    vendors['company_name'] = vendors['company_name'].replace(' ', '_', regex=True)

    vendor_cols_final = ['id',
                         'company_name',
                         'total_price',
                         'ceremony_decor',
                         'reception_decor',
                         'handhelds',
                         'size_of_wedding',
                         'location_Boulder',
                         'location_Denver',
                         'location_FootHills',
                         'location_Mountain_town_not_listed',
                         'location_Other',
                         'location_Summit_County',
                         'service_Delivery_no_service',
                         'service_full_service',
                         'service_Pickup']

    vendors.columns = vendor_cols_final

    def assign_clusters(row):
        florist_clusters = {'Flora_by_Nora':[0,2,4],
                'Madelyn_Claire_Floral_Design_&_Events': [3, 0, 1],
                'Little_Shop_of_Floral': [1, 3, 4],
                'Lumme_Creations': [6, 1, 2],
                'Rooted': [6, 4, 5],
                'Blush_&_Bay': [2,0,4]}
        cluster0, cluster1, cluster2, cluster3, cluster4, cluster5, cluster6 = 0,0,0,0,0,0,0
        if 0 in florist_clusters[row['company_name']]:
            cluster0 = 1
        if 1 in florist_clusters[row['company_name']]:
            cluster1 = 1
        if 2 in florist_clusters[row['company_name']]:
            cluster2 = 1
        if 3 in florist_clusters[row['company_name']]:
            cluster3 = 1
        if 4 in florist_clusters[row['company_name']]:
            cluster4 = 1
        if 5 in florist_clusters[row['company_name']]:
            cluster5 = 1
        if 6 in florist_clusters[row['company_name']]:
            cluster6 = 1
        return pd.Series({'cluster0':cluster0,
                   'cluster1':cluster1,
                   'cluster2':cluster2,
                   'cluster3':cluster3,
                   'cluster4':cluster4,
                   'cluster5':cluster5,
                   'cluster6':cluster6})

    vendors_clusters = vendors.apply(assign_clusters, axis=1)
    vendors_merged = pd.concat([vendors, vendors_clusters], axis=1)

    #Fill in 0 for Rooted NAs for now
    vendors_merged.fillna(0, inplace=True)

    features = ['total_price',
                'ceremony_decor',
                'reception_decor',
                'handhelds',
                'size_of_wedding',
                'location_Boulder',
                'location_Denver',
                'location_FootHills',
                'location_Mountain_town_not_listed',
                'location_Other',
                'location_Summit_County',
                'service_Delivery_no_service',
                'service_full_service',
                'service_Pickup',
                'cluster0',
                'cluster1',
                'cluster2',
                'cluster3',
                'cluster4',
                'cluster5',
                'cluster6']

    X = vendors_merged[features].values
    ss = StandardScaler()
    X = ss.fit_transform(X)

    total_price = float(request.form['total_price'])

    ceremony_decor, reception_decor, handhelds = 0,0,0
    if 'ceremony_decor' in request.form.keys():
        ceremony_decor = 1
    if 'reception_decor' in request.form.keys():
        reception_decor = 1
    if 'handhelds' in request.form.keys():
        handhelds = 1

    size_of_wedding = int(request.form['size_of_wedding'])

    location_Boulder, location_Denver, location_FootHills, location_Mountain_town_not_listed, location_Other, location_Summit_County = 0,0,0,0,0,0
    if request.form['location'] == 'location_Boulder':
        location_Boulder = 1
    if request.form['location'] == 'location_Denver':
        location_Denver = 1
    if request.form['location'] == 'location_FootHills':
        location_FootHills = 1
    if request.form['location'] == 'location_Mountain_town_not_listed':
        location_Mountain_town_not_listed = 1
    if request.form['location'] == 'location_Other':
        location_Other = 1
    if request.form['location'] == 'location_Summit_County':
        location_Summit_County = 1


    service_Delivery_no_service, service_full_service, service_Pickup = 0,0,0
    if request.form['service'] == 'service_Delivery_no_service':
        service_Delivery_no_service = 1
    if request.form['service'] == 'service_full_service':
        service_full_service = 1
    if request.form['service'] == 'service_Pickup':
        service_Pickup = 1


    cluster0, cluster1, cluster2,cluster3,cluster4, cluster5, cluster6 = 0,0,0,0,0,0,0
    if 'cluster0' in request.form.keys():
        cluster0 = 1
    if 'cluster1' in request.form.keys():
        cluster1 = 1
    if 'cluster2' in request.form.keys():
        cluster2 = 1
    if 'cluster3' in request.form.keys():
        cluster3 = 1
    if 'cluster4' in request.form.keys():
        cluster4 = 1
    if 'cluster5' in request.form.keys():
        cluster5 = 1
    if 'cluster6' in request.form.keys():
        cluster6 = 1

    example = pd.DataFrame({
                        'total_price':total_price,
                        'ceremony_decor':ceremony_decor,
                        'reception_decor':reception_decor,
                        'handhelds':handhelds,
                        'size_of_wedding':size_of_wedding,
                        'location_Boulder':location_Boulder,
                        'location_Denver':location_Denver,
                        'location_FootHills':location_FootHills,
                        'location_Mountain_town_not_listed':location_Mountain_town_not_listed,
                        'location_Other':location_Other,
                        'location_Summit_County':location_Summit_County,
                        'service_Delivery_no_service':service_Delivery_no_service,
                        'service_full_service':service_full_service,
                        'service_Pickup':service_Pickup, 'cluster0':cluster0, 'cluster1':cluster1, 'cluster2':cluster2, 'cluster3':cluster3, 'cluster4':cluster4, 'cluster5':cluster5, 'cluster6':cluster6}, index=[0])
    index_name = vendors_merged['company_name']
    cos_sims = cos_sim_recommendations(example, X, index_name, n=1)

    florist_info = {
    'Flora_by_Nora':
        {'name':'Flora by Nora', 'img_src':'/static/img/flora_by_nora.png', 'link':'https://www.florabynora.com/'},
    'Madelyn_Claire_Floral_Design_&_Events':
        {'name':'Madlyn Claire Floral Design', 'img_src':'/static/img/madelyn_claire.png', 'link':'https://madelynclairefloraldesign.com/'},
    'Little_Shop_of_Floral':
        {'name':'Little Shop of Floral', 'img_src':'/static/img/little_shop_of_floral.png', 'link':'https://www.littleshopoffloral.com/'},
    'Lumme_Creations':
        {'name':'Lumme Creations', 'img_src':'/static/img/lumme.png', 'link':'https://www.lummecreations.com/'},
    'Rooted':
        {'name':'Rooted Floral and Design', 'img_src':'/static/img/rooted.png', 'link':'https://www.rootedfloralanddesign.com/'},
    'Blush_&_Bay':
        {'name':'Blush and Bay', 'img_src':'/static/img/blush_and_bay.png', 'link':'http://www.blushandbay.com/'}}


    return render_template('recommendations.html', cos_sims = cos_sims, florist_info = florist_info)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
