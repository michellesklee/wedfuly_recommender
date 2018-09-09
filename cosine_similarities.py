import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import StandardScaler
pd.options.display.max_columns=25

vendors = pd.read_csv('../data/florists_updated.csv')

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


def cos_sim_recommendations(new_data, df, index_name, n=2):
    cs = cosine_similarity(new_data, df)
    rec_index = np.argsort(cs)[0][-n-1:][::-1][1:]
    recommendations = []
    for rec in rec_index:
        recommendations.append(index_name[rec])
    return recommendations



if __name__ == '__main__':

    example = pd.DataFrame({'total_price':1000.0,
                            'ceremony_decor':1.0,
                            'reception_decor':0.0,
                            'handhelds':1.0,
                            'size_of_wedding':300,
                            'location_Boulder':1.0,
                            'location_Denver':0.0,
                            'location_FootHills':0.0,
                            'location_Mountain_town_not_listed':0.0,
                            'location_Other':1.0,
                            'location_Summit_County':1.0,
                            'service_Delivery_full_service':1.0,
                            'service_Delivery_no_service':0.0,
                            'service_Drop_Off':0.0,
                            'cluster0':1, 'cluster1':0, 'cluster2':1, 'cluster3':0, 'cluster4':1, 'cluster5':0, 'cluster6':1}, index=[1])
    index_name = vendors_merged['company_name']

    cos_sim_recommendations(example, X, index_name, n=2)


# Top blush_and_bay labels: 2, 0, 4
# Top flora_by_nora labels: 0, 2, 4
# Top little_shop_of_floral labels: 1, 3, 4
# Top lumme labels: [6, 1, 2]
# Top madelyn_claire labels: 3, 0, 1
# Top rooted labels: 6, 4, 5
