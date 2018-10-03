from flask import Flask, render_template, request
import requests
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import StandardScaler
pd.options.display.max_columns=25

def dummy_vars(df, col_lst):
    for col in col_lst:
        df[col].fillna(0, inplace=True)
        df[col].replace(col, 1, inplace=True)

def cos_sim_recommendations(new_data, df, index_name, n=1):
    cs = cosine_similarity(new_data, df)
    rec_index = np.argsort(cs)[0][-n-1:][::-1][1:]
    recommendations = []
    for rec in rec_index:
        recommendations.append(index_name[rec])
    return recommendations

def assign_clusters(row):
    florist_clusters = {'Flora by Nora':[5,6,3],
            'Madelyn Claire Floral Design & Events': [4,0,3],
            'Little Shop of Floral': [4,0,6],
            'Lumme Creations': [4,5,6],
            'Rooted': [3,5,6],
            'Blush & Bay': [4,0,2]}
    cluster0, cluster1, cluster2, cluster3, cluster4, cluster5, cluster6 = 0,0,0,0,0,0,0
    if 0 in florist_clusters[row['Company_Name']]:
        cluster0 = 1
    if 1 in florist_clusters[row['Company_Name']]:
        cluster1 = 1
    if 2 in florist_clusters[row['Company_Name']]:
        cluster2 = 1
    if 3 in florist_clusters[row['Company_Name']]:
        cluster3 = 1
    if 4 in florist_clusters[row['Company_Name']]:
        cluster4 = 1
    if 5 in florist_clusters[row['Company_Name']]:
        cluster5 = 1
    if 6 in florist_clusters[row['Company_Name']]:
        cluster6 = 1
    return pd.Series({'cluster0':cluster0,
               'cluster1':cluster1,
               'cluster2':cluster2,
               'cluster3':cluster3,
               'cluster4':cluster4,
               'cluster5':cluster5,
               'cluster6':cluster6})

def assign_clusters_photos(row):
    photog_clusters = {'Allison Dobbs Photography':[0,2],
    'Evan Louis Photo':[0,1],
    'Rae Marie Photography':[0,1]}
    cluster0, cluster1, cluster2 = 0,0,0
    if 0 in photog_clusters[row['Company_Name']]:
        cluster0 = 1
    if 1 in photog_clusters[row['Company_Name']]:
        cluster1 = 1
    if 2 in photog_clusters[row['Company_Name']]:
        cluster2 = 1
    return pd.Series({'cluster0':cluster0,
               'cluster1':cluster1,
               'cluster2':cluster2})

app = Flask(__name__, static_url_path='/static')

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/florist_recommender', methods=['GET', 'POST'])
def recommender():
    return render_template('florist_recommender.html')

@app.route('/florist_recommendations', methods=['GET', 'POST'])
def florist_recommendations():
    vendors = pd.read_csv('data/florists_updated.csv')
    vendors = vendors.iloc[:, 1:11]
    vendors = vendors.drop(['Venue of wedding', 'Date of wedding'], axis=1)

    vendors = pd.get_dummies(vendors, columns=['Location', 'Service'])
    dummy_vars(vendors, ['Ceremony decor', 'Reception decor', 'Handhelds (bouquets and boutonnieres)'])
    vendors.columns = vendors.columns.str.replace(' ', '_')

    vendors_clusters = vendors.apply(assign_clusters, axis=1)
    vendors_merged = pd.concat([vendors, vendors_clusters], axis=1)

    vendors_merged.columns = ['company_name',
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
                     'service_Pickup', 'cluster0', 'cluster1', 'cluster2', 'cluster3', 'cluster4', 'cluster5', 'cluster6']

    features_std = vendors_merged.iloc[:, 1:].values
    ss = StandardScaler().fit(features_std)
    features_std = ss.transform(features_std)

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

    input_data = pd.DataFrame({
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
                        'service_Pickup':service_Pickup,
                        'cluster0':cluster0,
                        'cluster1':cluster1,
                        'cluster2':cluster2,
                        'cluster3':cluster3,
                        'cluster4':cluster4,
                        'cluster5':cluster5,
                        'cluster6':cluster6}, index=[0])

    index_name = vendors_merged['company_name']
    survey_data = input_data.values
    survey_data = ss.transform(survey_data)

    #change
    weights = np.array([5, 5, 5, 5, 3, 1, 1, 1, 1, 1, 1, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1])

    cos_sims = cos_sim_recommendations(survey_data*weights, features_std*weights, index_name, n=2)

    florist_info = {
    'Madelyn Claire Floral Design & Events':
        {'name':'Madlyn Claire Floral Design', 'img_src':'/static/img/madelyn_claire.png', 'link':'https://madelynclairefloraldesign.com/'},
    'Little Shop of Floral':
        {'name':'Little Shop of Floral', 'img_src':'/static/img/little_shop_of_floral.png', 'link':'https://www.littleshopoffloral.com/'},
    'Lumme Creations':
        {'name':'Lumme Creations', 'img_src':'/static/img/lumme.png', 'link':'https://www.lummecreations.com/'},
    'Rooted':
        {'name':'Rooted Floral and Design', 'img_src':'/static/img/rooted.png', 'link':'https://www.rootedfloralanddesign.com/'},
    'Blush & Bay':
        {'name':'Blush and Bay', 'img_src':'/static/img/blush_and_bay.png', 'link':'http://www.blushandbay.com/'},
    'Flora by Nora':
        {'name':'Flora by Nora', 'img_src':'/static/img/flora_by_nora.png', 'link':'https://www.florabynora.com/'}}
    return render_template('florist_recommendations.html', cos_sims = cos_sims, florist_info = florist_info)

@app.route('/photog_recommender', methods=['GET', 'POST'])
def photog_recommender():
    return render_template('photog_recommender.html')

@app.route('/photog_recommendations', methods=['GET', 'POST'])
def photog_recommendations():
    photogs = pd.read_csv('data/photog_data.csv')
    photogs = photogs.iloc[:, 1:10]
    photogs = photogs.drop(['Date of wedding', 'Location (City/Town)', 'Venue of Wedding'], axis=1)
    photogs = pd.get_dummies(photogs, columns=['Type of Coverage', 'Engagement Session'])
    photogs.columns = photogs.columns.str.replace(' ', '_')

    photogs_clusters = photogs.apply(assign_clusters_photos, axis=1)
    photogs_merged = pd.concat([photogs, photogs_clusters], axis = 1)

    photogs_merged.columns = ['company_name', 'total_price', 'shooters', 'size_of_wedding', 'type_of_coverage_over10_hours',
       'type_of_coverage_8_hours', 'type_of_coverage_elopement',
       'engagement_session_no', 'engagement_session_yes_included',
       'engagement_session_yes_paid_extra', 'cluster0', 'cluster1', 'cluster2']

    photogs_std = photogs_merged.iloc[:, 1:].values
    ss = StandardScaler().fit(photogs_std)
    photogs_std = ss.transform(photogs_std)

    total_price = float(request.form['total_price'])
    shooters = int(request.form['shooters'])
    size_of_wedding = int(request.form['size_of_wedding'])

    type_of_coverage_over10_hours, type_of_coverage_8_hours, type_of_coverage_elopement = 0,0,0
    if request.form['coverage'] == 'type_of_coverage_over10_hours':
        type_of_coverage_over10_hours = 1
    if request.form['coverage'] == 'type_of_coverage_8_hours':
        type_of_coverage_8_hours = 1
    if request.form['coverage'] == 'type_of_coverage_elopement':
        type_of_coverage_elopement = 1

    engagement_session_no, engagement_session_yes_included, engagement_session_yes_paid_extra = 0,0,0
    if request.form['engagement'] == 'engagement_session_no':
        engagement_session_no = 1
    if request.form['engagement'] == 'engagement_session_yes_included':
        engagement_session_yes_included = 1
    if request.form['engagement'] == 'engagement_session_yes_paid_extra':
        engagement_session_yes_paid_extra = 1

    cluster0, cluster1, cluster2 = 0,0,0
    if 'cluster0' in request.form.keys():
        cluster0 = 1
    if 'cluster1' in request.form.keys():
        cluster1 = 1
    if 'cluster2' in request.form.keys():
        cluster2 = 1

    input_data = pd.DataFrame({'total_price':total_price,
                               'shooters':shooters,
                               'size_of_wedding': size_of_wedding,
                               'type_of_coverage_over10_hours': type_of_coverage_over10_hours,
                               'type_of_coverage_8_hours':
                               type_of_coverage_8_hours,
                               'type_of_coverage_elopement': type_of_coverage_elopement,
                               'engagement_session_no': engagement_session_no,
                               'engagement_session_yes_included':
                               engagement_session_yes_included,
                               'engagement_session_yes_paid_extra': engagement_session_yes_paid_extra,
                               'cluster0': cluster0,
                               'cluster1': cluster1,
                               'cluster2': cluster2}, index=[0])
    index_name = photogs_merged['company_name']
    survey_data = input_data.values
    survey_data = ss.transform(survey_data)

    #new weights
    weights = np.array([6, 5, 5, 5, 5, 2, 2, 2, 3, 4, 4, 4])

    cos_sims = cos_sim_recommendations(survey_data*weights, photogs_std*weights, index_name, n=2)

    photogs_info = {'Allison Dobbs Photography':
    {'name':'Allison Dobbs Photography', 'img_src':'/static/img/allisondobbs.png',
    'link':'https://www.allisondobbsphotography.com/'}, 'Evan Louis Photo':{'name':'Evan Louis Photography',
    'img_src':'/static/img/evanlouis.png', 'link':'https://www.evanlouisphoto.com/'}, 'Rae Marie Photography':{'name':'Rae Marie Photography', 'img_src':'/static/img/raemarie.png', 'link':'https://www.raemariephotography.com/'}}

    return render_template('photog_recommendations.html', cos_sims = cos_sims, photogs_info = photogs_info)

# @app.route('/caterer_recommender', methods=['GET', 'POST'])
# def caterer_recommender():
#     return render_template('caterer_recommender.html')
#
# @app.route('/bakery_recommender', methods=['GET', 'POST'])
# def bakery_recommender():
#     return render_template('bakery_recommender.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
