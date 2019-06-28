# Perform LabelEncoder for app_info
info_labeler = LabelEncoder()
mask = ~app_info['category'].isnull()
app_info['category'][mask] = info_labeler.fit_transform(app_info['category'][mask])

class_num = len(np.unique(app_info['category']))
print('App Encode Class Number: %d' % class_num)

# Establish index dict
dictory = {}

for _, row in app_info.iterrows():
    if row['appid'] in dictory.keys():
        dictory[row['appid']].append(row['category'])
    else:
        dictory[row['appid']] = [row['category']]

print('App Info Index file is established~')  

# Build user_app_activated info tables
def get_app_encode(str_):
    active_index = []
    new_row = np.zeros(class_num)
    for ele in str_.split('#'):
        index = []
        if ele in dictory.keys():
            index = dictory[ele]
        if len(index) > 0:
            for ee in index:
                active_index.append(ee)
    new_row[active_index] = 1
                
    return new_row.tolist()
    
print('Target Reading Lines: %d' % len(app_package))

# According to the encode app_actived_lists, build app_install_features
verbose = 10000
start_time = time.time()
df_package = pd.DataFrame(columns=range(40))
df_package.insert(0, 'uid', [])

def build_frame():
    global df_package
    for index in range(len(app_package)):
        if index % verbose == 0:
            print('Build %d/%d, Cost Time: %f sec' 
                  % (index, len(app_package), time.time()-start_time))

        line = get_app_encode(app_package['appid'][index])
        line.insert(0, app_package['uid'][index])
        
        df_package.loc[index] = line

'''
build_frame()            

print(df_package)
'''